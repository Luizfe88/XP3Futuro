# 🔧 Correções para Habilitar Análise de Índices Futuros

## 📋 Problema Identificado
O bot está analisando **apenas ações** (PETR4, VALE3, ITUB4, etc.) e **não está detectando/analisando índices futuros** (WIN$, WDO$, IND$, etc.).

---

## 🎯 Correções Necessárias

### 1. **bot.py - Adicionar Futuros ao Pool de Análise**

#### **Localização:** Função `build_portfolio_and_top15()`

**Problema:**
- A função provavelmente está carregando apenas símbolos de ações do `config.ELITE_SYMBOLS` ou `config.SECTOR_MAP`
- Não há integração com os futuros detectados

**Correção:**
```python
def build_portfolio_and_top15():
    """Constrói portfólio incluindo AÇÕES + FUTUROS"""
    
    # 1. Carrega ações (código existente)
    stock_symbols = list(config.ELITE_SYMBOLS.keys())
    
    # 2. ✅ NOVO: Descobre e adiciona futuros ativos
    try:
        futures_map = utils.discover_all_futures()
        if futures_map:
            logger.info(f"📊 Futuros detectados: {list(futures_map.values())}")
            
            # Adiciona apenas futuros configurados
            for base in ['WIN', 'WDO', 'IND', 'WSP']:  # Bases desejadas
                if base in futures_map:
                    current_contract = futures_map[base]
                    stock_symbols.append(current_contract)
                    logger.info(f"✅ Adicionado futuro: {current_contract}")
    except Exception as e:
        logger.error(f"Erro ao descobrir futuros: {e}")
    
    # 3. Continua com análise normal...
    all_data = {}
    for symbol in stock_symbols:
        # ...análise de indicadores
```

---

### 2. **bot.py - Verificar Modo de Operação**

#### **Localização:** Função `main()` ou inicialização global

**Problema:**
- Variável `CURRENT_MODE` pode estar fixada em "AÇÕES"
- Não há detecção automática de futuros na inicialização

**Correção:**
```python
def main():
    global CURRENT_MODE
    
    # ✅ Detecta modo baseado em config ou disponibilidade
    if config.ENABLE_FUTURES_TRADING:
        # Tenta detectar futuros
        futures_map = utils.discover_all_futures()
        if futures_map:
            CURRENT_MODE = "HIBRIDO"  # Ações + Futuros
            logger.info(f"🔀 Modo HÍBRIDO ativado: {len(futures_map)} contratos futuros detectados")
        else:
            logger.warning("⚠️ ENABLE_FUTURES_TRADING=True mas nenhum futuro detectado")
            CURRENT_MODE = "AÇÕES"
    else:
        CURRENT_MODE = "AÇÕES"
    
    logger.info(f"📊 Modo de operação: {CURRENT_MODE}")
```

---

### 3. **utils.py - Função `discover_all_futures()`**

#### **Localização:** Verificar se existe e está funcional

**Problema:**
- Função pode não estar sendo chamada
- Pode estar falhando silenciosamente

**Correção:**
```python
def discover_all_futures(bases: list = None) -> dict:
    """
    Descobre contratos futuros ativos no MT5
    
    Returns:
        dict: {"WIN": "WING26", "WDO": "WDOG26", ...}
    """
    if bases is None:
        bases = ['WIN', 'WDO', 'IND', 'WSP', 'SMALL']
    
    mapping = {}
    
    if not futures_core:
        logger.warning("futures_core não disponível")
        return mapping
    
    try:
        manager = futures_core.get_manager()
        
        for base in bases:
            try:
                front = manager.find_front_month(base)
                if front:
                    mapping[base] = front
                    logger.info(f"✅ {base} → {front}")
                    
                    # ✅ Garante que está no Market Watch
                    if not mt5.symbol_select(front, True):
                        logger.warning(f"⚠️ Falha ao adicionar {front} ao Market Watch")
                else:
                    logger.warning(f"⚠️ Nenhum contrato front month encontrado para {base}")
            except Exception as e:
                logger.error(f"Erro ao detectar {base}: {e}")
        
        return mapping
        
    except Exception as e:
        logger.error(f"Erro fatal em discover_all_futures: {e}")
        return {}
```

---

### 4. **config.py - Adicionar Flag de Futuros**

#### **Localização:** Arquivo `config.py` (raiz do projeto)

**Problema:**
- Pode não existir flag para habilitar futuros
- Futuros podem estar desabilitados por padrão

**Correção:**
```python
# ============================================
# 🔄 CONFIGURAÇÃO DE FUTUROS
# ============================================

ENABLE_FUTURES_TRADING = True  # ✅ Habilita análise de futuros

# Bases de futuros para monitorar
FUTURES_BASES = ['WIN', 'WDO', 'IND']  # Mini Índice, Mini Dólar, Índice Cheio

# Peso de futuros no portfólio (0.0 a 1.0)
FUTURES_PORTFOLIO_WEIGHT = 0.30  # 30% do capital pode ser alocado em futuros
```

---

### 5. **futures_core.py - Melhorar Regex de Detecção**

#### **Localização:** Função `find_front_month()`

**Problema:**
- Regex pode não estar capturando corretamente os contratos
- MT5 `symbols_get()` pode não estar retornando resultados

**Correção:**
```python
def find_front_month(self, base_symbol):
    """Detecta contrato front month com fallback robusto"""
    
    # 1. Tenta busca específica com wildcards
    patterns = [
        f"{base_symbol}[FGHJKMNQUVXZ][0-9][0-9]",  # WING26, WDOG26
        f"{base_symbol}*",  # Fallback genérico
    ]
    
    candidates = []
    
    for pattern in patterns:
        try:
            symbols = self.mt5.symbols_get(group=f"*{pattern}*")
            if symbols:
                logger.debug(f"Encontrados {len(symbols)} símbolos com padrão {pattern}")
                candidates.extend(symbols)
                break  # Para no primeiro padrão que retornar resultados
        except Exception as e:
            logger.debug(f"Erro com padrão {pattern}: {e}")
    
    if not candidates:
        logger.warning(f"❌ Nenhum símbolo encontrado para {base_symbol}")
        # ✅ Tenta alternativa: construir código manualmente
        return self._try_manual_detection(base_symbol)
    
    # 2. Filtra e ordena por OI + vencimento
    # ... (resto do código existente)
```

**Adicionar função auxiliar:**
```python
def _try_manual_detection(self, base_symbol):
    """Tenta construir símbolo manualmente baseado no mês atual"""
    from datetime import datetime
    
    # Mapa de meses para letras de futuros
    month_codes = {
        1: 'F', 2: 'G', 3: 'H', 4: 'J', 5: 'K', 6: 'M',
        7: 'N', 8: 'Q', 9: 'U', 10: 'V', 11: 'X', 12: 'Z'
    }
    
    now = datetime.now()
    year_code = str(now.year)[-2:]  # Últimos 2 dígitos do ano
    
    # Tenta mês atual e próximos 3 meses
    for offset in range(4):
        month = ((now.month - 1 + offset) % 12) + 1
        code = month_codes.get(month, 'G')
        
        # Constrói símbolo: ex. WING26
        symbol = f"{base_symbol}{code}{year_code}"
        
        # Verifica se existe
        info = self.mt5.symbol_info(symbol)
        if info:
            logger.info(f"✅ Detecção manual bem-sucedida: {symbol}")
            return symbol
    
    logger.error(f"❌ Falha na detecção manual de {base_symbol}")
    return None
```

---

### 6. **bot.py - Garantir Chamada na Inicialização**

#### **Localização:** Função `main()` após `mt5.initialize()`

**Problema:**
- `discover_all_futures()` pode não estar sendo chamada
- Market Watch pode não estar sincronizado

**Correção:**
```python
def main():
    # ... (código de inicialização MT5)
    
    # ✅ GARANTE SINCRONIZAÇÃO COM FUTUROS
    logger.info("🔍 Descobrindo contratos futuros...")
    futures_mapping = utils.discover_all_futures()
    
    if futures_mapping:
        logger.info(f"✅ Futuros mapeados: {futures_mapping}")
        
        # Salva em variável global ou config
        global ACTIVE_FUTURES
        ACTIVE_FUTURES = futures_mapping
        
        # Adiciona ao Market Watch
        for symbol in futures_mapping.values():
            if mt5.symbol_select(symbol, True):
                logger.info(f"📊 {symbol} adicionado ao Market Watch")
            else:
                logger.warning(f"⚠️ Falha ao adicionar {symbol}")
    else:
        logger.warning("⚠️ Nenhum futuro detectado - operando apenas com ações")
    
    # ... (continua inicialização)
```

---

### 7. **utils.py - Função `is_future()` Correta**

#### **Localização:** Verificar implementação

**Problema:**
- Função pode não estar identificando corretamente futuros
- Lógica pode estar retornando `False` para contratos válidos

**Correção:**
```python
def is_future(symbol: str) -> bool:
    """
    Identifica se um símbolo é um contrato futuro
    
    Returns:
        bool: True se for futuro, False se for ação
    """
    if not symbol:
        return False
    
    symbol_upper = symbol.upper()
    
    # 1. Verifica se tem sufixo genérico ($)
    if '$' in symbol_upper:
        return True
    
    # 2. Verifica prefixos conhecidos + código de vencimento
    future_bases = ['WIN', 'WDO', 'IND', 'DOL', 'WSP', 'SMALL', 'DI1', 'ICF', 'CCM', 'BGI']
    
    for base in future_bases:
        if symbol_upper.startswith(base):
            # Verifica se tem código de mês (letra) + ano (2 dígitos)
            # Exemplo: WING26, WDOG26
            pattern = f"{base}[FGHJKMNQUVXZ][0-9]{{2}}"
            import re
            if re.match(pattern, symbol_upper):
                return True
    
    # 3. Fallback: Verifica no MT5 se tem data de expiração
    try:
        info = mt5.symbol_info(symbol)
        if info and hasattr(info, 'expiration_time'):
            if info.expiration_time > 0:
                return True
    except:
        pass
    
    return False
```

---

### 8. **Verificar Logs de Inicialização**

#### **O que procurar no terminal ao iniciar o bot:**

✅ **Logs esperados se estiver correto:**
```
✅ Conectado ao MT5 correto: C:\Program Files\...
🔍 Descobrindo contratos futuros...
✅ WIN → WING26
✅ WDO → WDOG26
✅ IND → INDG26
✅ Futuros mapeados: {'WIN': 'WING26', 'WDO': 'WDOG26', 'IND': 'INDG26'}
📊 WING26 adicionado ao Market Watch
📊 WDOG26 adicionado ao Market Watch
📊 INDG26 adicionado ao Market Watch
🔀 Modo HÍBRIDO ativado: 3 contratos futuros detectados
📊 Modo de operação: HIBRIDO
```

❌ **Logs que indicam problema:**
```
⚠️ ENABLE_FUTURES_TRADING=True mas nenhum futuro detectado
❌ Nenhum símbolo encontrado para WIN
futures_core não disponível
⚠️ Nenhum futuro detectado - operando apenas com ações
📊 Modo de operação: AÇÕES
```

---

## 🧪 Como Testar

### 1. **Teste Manual no Terminal Python:**
```python
import MetaTrader5 as mt5
import futures_core

mt5.initialize()

# Teste 1: Busca por wildcards
symbols = mt5.symbols_get(group="*WIN*")
print(f"Encontrados: {[s.name for s in symbols]}")

# Teste 2: Usa futures_core
manager = futures_core.get_manager()
front = manager.find_front_month("WIN")
print(f"Front month: {front}")

# Teste 3: Verifica Open Interest
info = mt5.symbol_info(front)
print(f"Open Interest: {info.session_open_interest}")
```

### 2. **Verificar Market Watch no MT5:**
- Abra o MetaTrader 5
- Vá em "Visualizar" → "Market Watch" (Ctrl+M)
- Procure por contratos: WING26, WDOG26, INDG26
- Se não aparecerem, use o botão direito → "Símbolos" → Procure por "WIN", "WDO"

---

## 📊 Checklist de Implementação

- [ ] **1.** Adicionar futuros no `build_portfolio_and_top15()`
- [ ] **2.** Implementar detecção de modo híbrido no `main()`
- [ ] **3.** Melhorar `discover_all_futures()` com fallbacks
- [ ] **4.** Adicionar `ENABLE_FUTURES_TRADING = True` no config
- [ ] **5.** Corrigir regex e wildcards no `find_front_month()`
- [ ] **6.** Adicionar detecção manual de contratos (`_try_manual_detection`)
- [ ] **7.** Garantir chamada de `discover_all_futures()` no `main()`
- [ ] **8.** Validar função `is_future()` com testes
- [ ] **9.** Verificar logs de inicialização (futuros detectados?)
- [ ] **10.** Testar manualmente no terminal Python

---

## 🔍 Diagnóstico Rápido

Execute este script para diagnosticar o problema:

```python
# diagnostico_futuros.py
import MetaTrader5 as mt5
import futures_core
import utils

print("="*60)
print("DIAGNÓSTICO DE FUTUROS")
print("="*60)

# 1. MT5 conectado?
if mt5.initialize():
    print("✅ MT5 conectado")
else:
    print("❌ MT5 NÃO conectado")
    exit()

# 2. Símbolos disponíveis?
win_symbols = mt5.symbols_get(group="*WIN*")
print(f"\n📊 Símbolos WIN encontrados: {len(win_symbols)}")
for s in win_symbols[:5]:
    print(f"   - {s.name}")

# 3. futures_core funcional?
try:
    manager = futures_core.get_manager()
    front_win = manager.find_front_month("WIN")
    print(f"\n✅ Front month WIN: {front_win}")
except Exception as e:
    print(f"\n❌ Erro no futures_core: {e}")

# 4. discover_all_futures funciona?
try:
    mapping = utils.discover_all_futures()
    print(f"\n✅ Mapeamento completo: {mapping}")
except Exception as e:
    print(f"\n❌ Erro no discover_all_futures: {e}")

# 5. is_future detecta corretamente?
test_symbols = ["WING26", "WDOG26", "PETR4", "WIN$"]
print("\n🧪 Testes is_future():")
for sym in test_symbols:
    result = utils.is_future(sym)
    print(f"   {sym}: {'✅ FUTURO' if result else '❌ AÇÃO'}")

print("="*60)
```

---

## 📝 Notas Importantes

1. **Corretora:** Alguns brokers têm nomenclaturas diferentes (ex: WINJ26 vs WING26)
2. **Horário:** Futuros fora do horário de pregão podem ter OI zerado
3. **Vencimento:** Contratos próximos ao vencimento (< 5 dias) podem não ser detectados
4. **Permissões:** Verifique se sua conta MT5 tem permissão para operar futuros

---

## ✅ Resultado Esperado

Após implementar as correções, o log deve mostrar análises de futuros:

```
📊 XP3 PRO - LOG DE ANÁLISES
📅 Janela: 29/01/2026 15:00–18:00
================================================================================

2026-01-29 15:10:32 | INFO | analysis | symbol=WING26 | signal=BUY | strategy=ELITE_V5.5 | score=85
2026-01-29 15:10:32 | INFO | analysis | symbol=WDOG26 | signal=SELL | strategy=ELITE_V5.5 | score=70
2026-01-29 15:10:33 | INFO | analysis | symbol=PETR4 | signal=BUY | strategy=ELITE_V5.5 | score=100
...
```

---

**Boa sorte com as correções! 🚀**
