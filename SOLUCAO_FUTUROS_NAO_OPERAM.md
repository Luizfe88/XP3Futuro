# 🚨 DIAGNÓSTICO: Bot Não Opera Índices Futuros (WIN/WDO)

## 🔍 PROBLEMA IDENTIFICADO

Após implementar todas as mudanças sugeridas, o bot **ainda não analisa nem opera índices futuros**. Este é um problema **crítico e distinto** dos 12 problemas anteriores.

---

## 🎯 CAUSA RAIZ

O bot possui um **sistema de descoberta automática de futuros** (`discover_all_futures()` em `utils.py` linha 394), mas há **4 falhas sequenciais** que impedem a operação:

### ❌ FALHA 1: Descoberta Não é Executada no Startup

**Localização:** `bot.py` linha 5894-5899

```python
# ✅ CORRETO (já existe)
try:
    fm = utils.discover_all_futures()
    if fm:
        logger.info(f"Futuros mapeados: {fm}")
except Exception as e:
    logger.warning(f"Erro ao descobrir futuros: {e}")
```

**Problema:** Este código **é executado**, mas:
1. Não valida se os símbolos foram realmente adicionados ao Market Watch
2. Não verifica se os símbolos estão em `SECTOR_MAP`
3. Não adiciona à lista de símbolos escaneados

---

### ❌ FALHA 2: Símbolos Genéricos em SECTOR_MAP

**Localização:** `config.py` linha 312-316

```python
SECTOR_MAP = {
    # ... ações normais
    "WING26": "FUTUROS",    # ❌ Símbolo hardcoded (vence em fev/26)
    "WDOG26": "FUTUROS",    # ❌ Símbolo hardcoded (vence em fev/26)
    "SMALL$": "FUTUROS",    # ❌ Genérico não resolvido
    "WSPH26": "FUTUROS",    # ❌ Símbolo hardcoded
    "BGIG26": "FUTUROS"     # ❌ Símbolo hardcoded
}
```

**Problema:**
- `WING26` e `WDOG26` **já expiraram** em Janeiro/2025
- Código genérico `SMALL$` permanece sem ser resolvido
- Função `discover_all_futures()` cria `ACTIVE_FUTURES`, mas **não atualiza SECTOR_MAP** corretamente

---

### ❌ FALHA 3: Fast Loop Não Escaneia Futuros

**Localização:** `bot.py` linha 5091-5101

```python
# ❌ PROBLEMA: usa apenas optimized_params.keys()
symbols_to_scan = list(optimized_params.keys())

# ✅ Tenta adicionar WIN/WDO, mas FALHA silenciosamente
current_win = utils.resolve_current_symbol("WIN")
current_wdo = utils.resolve_current_symbol("WDO")

if current_win and current_win not in symbols_to_scan:
    symbols_to_scan.append(current_win)  # ❌ Nunca acontece!
if current_wdo and current_wdo not in symbols_to_scan:
    symbols_to_scan.append(current_wdo)  # ❌ Nunca acontece!
```

**Por que falha:**
1. `resolve_current_symbol()` **NÃO EXISTE** em `utils.py`!
2. Função retorna `None` silenciosamente
3. Futuros nunca entram em `symbols_to_scan`

---

### ❌ FALHA 4: build_portfolio_and_top15() Ignora Futuros

**Localização:** `bot.py` linha 2500-2844

```python
def build_portfolio_and_top15():
    # Só usa ELITE_SYMBOLS + fallback para ações
    elite_path = config.ELITE_SYMBOLS_JSON_PATH
    
    # ... código que carrega apenas ações
    
    # ❌ NUNCA adiciona futuros descobertos automaticamente!
```

---

## 🔧 SOLUÇÃO COMPLETA

### PASSO 1: Criar `resolve_current_symbol()` em `utils.py`

**Adicionar APÓS linha 428 em `utils.py`:**

```python
def resolve_current_symbol(base: str) -> Optional[str]:
    """
    Resolve símbolo genérico (WIN, WDO, etc) para contrato atual.
    
    Exemplo:
        resolve_current_symbol("WIN") → "WINJ25" (em Janeiro/2025)
        resolve_current_symbol("WDO") → "WDOF25" (em Janeiro/2025)
    
    Returns:
        Símbolo específico do contrato ativo, ou None se não encontrado
    """
    try:
        # 1. Checa cache em config.ACTIVE_FUTURES
        generic = f"{base}$"
        active_futures = getattr(config, "ACTIVE_FUTURES", {})
        
        if generic in active_futures:
            cached_symbol = active_futures[generic]
            
            # Valida se ainda é válido (não expirou)
            info = mt5.symbol_info(cached_symbol)
            if info:
                # Verifica data de expiração
                exp_time = getattr(info, "expiration_time", None)
                if exp_time:
                    from datetime import datetime
                    if isinstance(exp_time, datetime):
                        if exp_time > datetime.now():
                            # Ainda válido
                            logger.debug(f"✅ {base}: Cache hit → {cached_symbol}")
                            return cached_symbol
                        else:
                            logger.warning(f"⚠️ {base}: Contrato expirado {cached_symbol} (exp: {exp_time})")
                else:
                    # Sem exp_time = contrato válido
                    return cached_symbol
        
        # 2. Cache miss ou expirado - redescobre
        logger.info(f"🔍 {base}: Redescoberta necessária...")
        
        # Busca candidatos
        candidates = get_futures_candidates(base)
        
        if not candidates:
            logger.error(f"❌ {base}: Nenhum candidato encontrado!")
            return None
        
        # Ordena por score (melhor = mais líquido + mais distante)
        candidates_sorted = sorted(
            candidates, 
            key=lambda c: (-calculate_contract_score(c), c.get("days_to_exp", 9999))
        )
        
        best = candidates_sorted[0]
        best_symbol = best.get("symbol")
        
        if not best_symbol:
            logger.error(f"❌ {base}: Melhor candidato sem símbolo!")
            return None
        
        # 3. Valida que símbolo está disponível no MT5
        if not mt5.symbol_select(best_symbol, True):
            logger.error(f"❌ {base}: Não foi possível selecionar {best_symbol} no MT5")
            return None
        
        # 4. Atualiza cache
        active_futures[generic] = best_symbol
        setattr(config, "ACTIVE_FUTURES", active_futures)
        
        # 5. Atualiza SECTOR_MAP
        sector_map = getattr(config, "SECTOR_MAP", {})
        sector_map[best_symbol] = "FUTUROS"
        setattr(config, "SECTOR_MAP", sector_map)
        
        logger.info(
            f"✅ {base}: Resolvido para {best_symbol} | "
            f"Expira em {best.get('days_to_exp', 0)} dias | "
            f"Volume: {best.get('volume', 0):,.0f}"
        )
        
        return best_symbol
        
    except Exception as e:
        logger.error(f"❌ Erro ao resolver {base}: {e}", exc_info=True)
        return None
```

---

### PASSO 2: Atualizar `discover_all_futures()` em `utils.py`

**SUBSTITUIR função existente (linha 394-429):**

```python
def discover_all_futures() -> dict:
    """
    Descobre todos os contratos futuros ativos da B3.
    
    Atualiza automaticamente:
    - config.ACTIVE_FUTURES (mapeamento genérico → específico)
    - config.SECTOR_MAP (adiciona futuros descobertos)
    
    Returns:
        Dict com mapeamentos (ex: {"WIN$": "WINJ25", "WDO$": "WDOF25"})
    """
    try:
        logger.info("🔍 Iniciando Auto-Discovery de Contratos Futuros...")
        broker = detect_broker()
        logger.info(f"📡 Corretora detectada: {broker}")
        
        # Bases para descobrir
        generics = ["WIN$", "WDO$", "SMALL$", "WSP$"]
        
        result = {}
        sector_map = getattr(config, "SECTOR_MAP", {})
        
        for generic in generics:
            base = generic.replace("$", "")
            
            logger.info(f"\n🎯 Descobrindo {generic}...")
            
            # Usa resolve_current_symbol (que já implementamos)
            specific_symbol = resolve_current_symbol(base)
            
            if specific_symbol:
                result[generic] = specific_symbol
                
                # Atualiza SECTOR_MAP
                sector_map[specific_symbol] = "FUTUROS"
                
                # Remove entrada genérica antiga se existir
                if generic in sector_map:
                    sector_map.pop(generic, None)
                
                logger.info(f"   ✅ {generic} → {specific_symbol}")
            else:
                # Fallback manual
                fallback = _fallback_future_symbol(base)
                if fallback:
                    logger.warning(f"   ⚠️ {generic} → {fallback} (FALLBACK)")
                    result[generic] = fallback
                    sector_map[fallback] = "FUTUROS"
                else:
                    logger.error(f"   ❌ {generic}: FALHA TOTAL (sem fallback)")
        
        # Salva resultados
        if result:
            setattr(config, "ACTIVE_FUTURES", result)
            setattr(config, "SECTOR_MAP", sector_map)
            
            # Salva em arquivo para auditoria
            try:
                out_dir = Path("futures_optimizer_output")
                out_dir.mkdir(parents=True, exist_ok=True)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = out_dir / f"futures_mappings_{timestamp}.json"
                
                metadata = {
                    "timestamp": datetime.now().isoformat(),
                    "broker": broker,
                    "mappings": result,
                    "sector_map_updated": True
                }
                
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2, default=str)
                
                logger.info(f"💾 Mapeamentos salvos em: {filepath}")
            except Exception as e:
                logger.warning(f"⚠️ Erro ao salvar mapeamentos: {e}")
        
        logger.info(f"\n📊 Resumo Discovery:")
        logger.info(f"   ✅ {len(result)}/{len(generics)} contratos descobertos")
        logger.info(f"   📋 Mapeamentos: {result}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Erro grave no discover_all_futures: {e}", exc_info=True)
        return {}
```

---

### PASSO 3: Forçar Redescobrimento no Fast Loop

**SUBSTITUIR bloco em `bot.py` linha 5091-5101:**

```python
# ============================================
# 7️⃣ PROCESSAMENTO DE SINAIS (SE PERMITIDO)
# ============================================
if market_status["new_entries_allowed"]:
    # Base: símbolos otimizados (ações)
    symbols_to_scan = list(optimized_params.keys())
    
    # ✅ CRÍTICO: Adiciona futuros descobertos dinamicamente
    try:
        active_futures = getattr(config, "ACTIVE_FUTURES", {})
        
        # Se cache vazio, força descoberta
        if not active_futures:
            logger.warning("🔄 Cache de futuros vazio - executando descoberta...")
            active_futures = utils.discover_all_futures()
        
        # Adiciona futuros à lista de scan
        for generic, specific_symbol in active_futures.items():
            if specific_symbol and specific_symbol not in symbols_to_scan:
                # Valida horário específico de futuros
                if utils.is_time_allowed_for_symbol(specific_symbol, "FUTUROS"):
                    symbols_to_scan.append(specific_symbol)
                    logger.debug(f"✅ Futuro adicionado ao scan: {specific_symbol}")
        
        logger.info(
            f"📋 Símbolos no scan: {len(symbols_to_scan)} "
            f"(Ações: {len(optimized_params)}, Futuros: {len(active_futures)})"
        )
        
    except Exception as e:
        logger.error(f"❌ Erro ao adicionar futuros: {e}", exc_info=True)
    
    # Continua com lógica normal de scan
    for sym in symbols_to_scan:
        ind_data = bot_state.get_indicators(sym)
        
        if not ind_data or ind_data.get("error"):
            continue
        
        # ... resto do código
```

---

### PASSO 4: Adicionar Futuros ao build_portfolio_and_top15()

**ADICIONAR ao final da função `build_portfolio_and_top15()` (linha ~2836):**

```python
def build_portfolio_and_top15():
    # ... código existente de ações
    
    # ✅ NOVO: Adiciona futuros ao final
    try:
        active_futures = getattr(config, "ACTIVE_FUTURES", {})
        
        for generic, specific_symbol in active_futures.items():
            if not specific_symbol:
                continue
            
            # Verifica se já foi adicionado (evita duplicatas)
            if specific_symbol in indicators:
                continue
            
            # Calcula indicadores para futuro
            try:
                df = utils.safe_copy_rates(specific_symbol, mt5.TIMEFRAME_M5, 100)
                
                if df is None or len(df) < 50:
                    logger.debug(f"⚠️ {specific_symbol}: Dados insuficientes")
                    continue
                
                # Indicadores simplificados para futuros (M5)
                ind = utils.quick_indicators_custom(specific_symbol, mt5.TIMEFRAME_M5, df=df)
                
                if ind and not ind.get("error"):
                    # Calcula score
                    score = utils.calculate_signal_score(ind)
                    
                    # Adiciona ao pool
                    scored.append((score, specific_symbol))
                    indicators[specific_symbol] = ind
                    
                    logger.debug(
                        f"✅ Futuro adicionado: {specific_symbol} | Score: {score:.0f}"
                    )
                
            except Exception as e:
                logger.error(f"❌ Erro ao processar {specific_symbol}: {e}")
                continue
        
    except Exception as e:
        logger.error(f"❌ Erro ao adicionar futuros no build_portfolio: {e}")
    
    # Re-ordena scored incluindo futuros
    scored.sort(reverse=True, key=lambda x: x[0])
    selected_top = [s for _, s in scored[:15]]
    
    bot_state.update(indicators, selected_top)
    update_bot_bridge()
    
    return indicators, selected_top
```

---

### PASSO 5: Atualizar SECTOR_MAP no Startup

**ADICIONAR em `bot.py` após linha 5899:**

```python
try:
    fm = utils.discover_all_futures()
    if fm:
        logger.info(f"Futuros mapeados: {fm}")
        
        # ✅ NOVO: Valida que SECTOR_MAP foi atualizado
        sector_map = config.SECTOR_MAP
        futuros_no_sector = [s for s in fm.values() if s in sector_map]
        
        logger.info(
            f"📊 Validação SECTOR_MAP: "
            f"{len(futuros_no_sector)}/{len(fm)} futuros registrados"
        )
        
        if len(futuros_no_sector) < len(fm):
            logger.warning(
                f"⚠️ Alguns futuros não estão no SECTOR_MAP: "
                f"{set(fm.values()) - set(futuros_no_sector)}"
            )
        
except Exception as e:
    logger.warning(f"Erro ao descobrir futuros: {e}")
```

---

### PASSO 6: Limpar Símbolos Hardcoded Expirados

**EDITAR `config.py` linha 312-316:**

```python
SECTOR_MAP = {
    # ... todas as ações normais
    
    # ❌ REMOVER LINHAS ANTIGAS:
    # "WING26": "FUTUROS",  # EXPIRADO
    # "WDOG26": "FUTUROS",  # EXPIRADO
    # "SMALL$": "FUTUROS",  # GENÉRICO
    # "WSPH26": "FUTUROS",  # EXPIRADO
    # "BGIG26": "FUTUROS"   # EXPIRADO
    
    # ✅ Futuros serão adicionados dinamicamente via discover_all_futures()
}
```

---

### PASSO 7: Adicionar Log de Debug

**ADICIONAR função de diagnóstico em `utils.py`:**

```python
def diagnose_futures_status() -> dict:
    """
    Diagnóstico completo do status dos futuros.
    Útil para debug quando futuros não operam.
    """
    result = {
        "active_futures_cache": getattr(config, "ACTIVE_FUTURES", {}),
        "futures_in_sector_map": [],
        "market_watch_status": {},
        "data_availability": {}
    }
    
    try:
        # 1. Futuros no SECTOR_MAP
        sector_map = getattr(config, "SECTOR_MAP", {})
        result["futures_in_sector_map"] = [
            sym for sym, sector in sector_map.items() 
            if sector == "FUTUROS"
        ]
        
        # 2. Status no Market Watch
        for symbol in result["futures_in_sector_map"]:
            info = mt5.symbol_info(symbol)
            result["market_watch_status"][symbol] = {
                "exists": info is not None,
                "visible": getattr(info, "visible", False) if info else False,
                "select": getattr(info, "select", False) if info else False,
                "expiration": getattr(info, "expiration_time", None) if info else None
            }
        
        # 3. Disponibilidade de dados
        for symbol in result["futures_in_sector_map"]:
            try:
                rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 10)
                result["data_availability"][symbol] = {
                    "has_data": rates is not None and len(rates) > 0,
                    "bars_available": len(rates) if rates else 0
                }
            except Exception as e:
                result["data_availability"][symbol] = {
                    "has_data": False,
                    "error": str(e)
                }
        
        # 4. Log formatado
        logger.info("="*60)
        logger.info("🔍 DIAGNÓSTICO DE FUTUROS")
        logger.info("="*60)
        
        logger.info(f"\n1️⃣ Cache ACTIVE_FUTURES:")
        for k, v in result["active_futures_cache"].items():
            logger.info(f"   {k} → {v}")
        
        logger.info(f"\n2️⃣ SECTOR_MAP ({len(result['futures_in_sector_map'])} futuros):")
        for sym in result["futures_in_sector_map"]:
            status = result["market_watch_status"].get(sym, {})
            logger.info(
                f"   {sym}: "
                f"Existe={status.get('exists')} | "
                f"Visível={status.get('visible')} | "
                f"Selecionado={status.get('select')}"
            )
        
        logger.info(f"\n3️⃣ Disponibilidade de Dados:")
        for sym, data in result["data_availability"].items():
            logger.info(
                f"   {sym}: "
                f"Dados={'OK' if data.get('has_data') else 'FALHA'} | "
                f"Barras={data.get('bars_available', 0)}"
            )
        
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Erro no diagnóstico: {e}", exc_info=True)
    
    return result
```

**Chamar no startup (bot.py após linha 5899):**

```python
try:
    fm = utils.discover_all_futures()
    if fm:
        logger.info(f"Futuros mapeados: {fm}")
        
        # ✅ Executa diagnóstico
        utils.diagnose_futures_status()
        
except Exception as e:
    logger.warning(f"Erro ao descobrir futuros: {e}")
```

---

## 📊 CHECKLIST DE VALIDAÇÃO

Após implementar todos os passos, verificar:

- [ ] ✅ `discover_all_futures()` retorna dict não-vazio
- [ ] ✅ `config.ACTIVE_FUTURES` contém mapeamentos (ex: `{"WIN$": "WINJ25"}`)
- [ ] ✅ `config.SECTOR_MAP` contém símbolos específicos (ex: `"WINJ25": "FUTUROS"`)
- [ ] ✅ `diagnose_futures_status()` mostra "Dados=OK" para todos futuros
- [ ] ✅ Fast loop loga "Futuro adicionado ao scan: WINJ25"
- [ ] ✅ `build_portfolio_and_top15()` inclui futuros no indicators dict
- [ ] ✅ Dashboard mostra futuros no TOP15
- [ ] ✅ Logs mostram análise de futuros (RSI, ADX, etc)
- [ ] ✅ Bot entra em posição de futuro quando sinal válido

---

## 🎯 LOG ESPERADO (SUCESSO)

```log
[2025-01-28 09:05:00] INFO: 🔍 Iniciando Auto-Discovery de Contratos Futuros...
[2025-01-28 09:05:00] INFO: 📡 Corretora detectada: Clear
[2025-01-28 09:05:00] INFO: 🎯 Descobrindo WIN$...
[2025-01-28 09:05:01] INFO:    Candidatos: WINJ25, WING25, WINM25
[2025-01-28 09:05:01] INFO:    ✅ WIN$ → WINJ25 (exp: 45 dias, vol: 1.2M)
[2025-01-28 09:05:02] INFO: 🎯 Descobrindo WDO$...
[2025-01-28 09:05:03] INFO:    ✅ WDO$ → WDOF25 (exp: 38 dias, vol: 850K)
[2025-01-28 09:05:04] INFO: 📊 Resumo Discovery:
[2025-01-28 09:05:04] INFO:    ✅ 2/4 contratos descobertos
[2025-01-28 09:05:04] INFO:    📋 Mapeamentos: {'WIN$': 'WINJ25', 'WDO$': 'WDOF25'}
[2025-01-28 09:05:05] INFO: 💾 Mapeamentos salvos em: futures_optimizer_output/futures_mappings_20250128_090505.json

[2025-01-28 09:05:10] INFO: 📋 Símbolos no scan: 47 (Ações: 45, Futuros: 2)
[2025-01-28 09:05:15] INFO: ✅ Futuro adicionado: WINJ25 | Score: 68.5
[2025-01-28 09:05:20] INFO: 🚀 ENVIANDO ENTRADA BUY em WINJ25 | Vol: 2 @ 134500.00
```

---

## ⚠️ ERROS COMUNS E SOLUÇÕES

### Erro: "TypeError: resolve_current_symbol() got an unexpected keyword argument"

**Causa:** Chamando `resolve_current_symbol()` com parâmetros errados

**Solução:** Usar apenas `resolve_current_symbol("WIN")` (sem parâmetros extras)

---

### Erro: "KeyError: 'WIN$' not in ACTIVE_FUTURES"

**Causa:** `discover_all_futures()` não foi executado ou falhou

**Solução:** Verificar logs no startup. Se não aparecer "Iniciando Auto-Discovery", adicionar chamada manual:

```python
# No main(), antes do fast_loop
active_futures = utils.discover_all_futures()
if not active_futures:
    logger.critical("❌ FALHA CRÍTICA: Nenhum futuro descoberto!")
```

---

### Erro: "No data available for WINJ25"

**Causa:** Símbolo não está no Market Watch do MT5

**Solução:** Forçar adição:

```python
if not mt5.symbol_select("WINJ25", True):
    logger.error("Não foi possível adicionar WINJ25 ao Market Watch")
```

---

## 🚀 RESULTADO ESPERADO

Após implementação completa:

1. **Startup**: Bot descobre WIN, WDO, SMALL automaticamente
2. **Cada Ciclo**: Futuros são analisados em M5 (mais rápido que ações)
3. **Sinais**: Bot entra em futuros quando RSI+ADX+VWAP alinham
4. **Logs**: Mostram análise contínua de futuros
5. **Dashboard**: Exibe futuros no TOP15 e posições abertas

**Tempo estimado de implementação:** 2-3 horas  
**Complexidade:** Média (requer testes manuais no MT5)  
**Impacto:** 🔴 CRÍTICO (sem isso, bot opera apenas 50% do potencial)

---

**Última Atualização:** 28/01/2026  
**Versão:** 1.0  
**Status:** Pronto para implementação
