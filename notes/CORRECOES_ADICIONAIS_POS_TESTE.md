# 🚨 DIAGNÓSTICO CRÍTICO - Resultado do Teste

## 📊 ANÁLISE DO OUTPUT (16:24:47)

### ❌ PROBLEMA #1: SÍMBOLO ERRADO
```
SIMBOLO: WING26  ❌ ERRADO
Esperado: WIN$N  ✓ CORRETO
```

**O que aconteceu:**
- Você executou: `--symbols 'WIN$N'`
- Sistema carregou: `WING26` (contrato de vencimento G26 = Fevereiro 2026)

**Por que isso é um problema:**
1. **Dados fragmentados**: WING26 tem apenas ~30 dias de vida útil
2. **Rollover não tratado**: Quando vencer, você não terá dados históricos
3. **Série contínua $N**: Deveria concatenar automaticamente WIN_F26, WIN_G26, etc.

**SOLUÇÃO IMEDIATA:**

Arquivo: `otimizador_semanal.py` - Linha ~152

Adicione ANTES de carregar dados do MT5:

```python
def normalize_futures_symbol(symbol: str) -> str:
    """
    Converte símbolo genérico para série contínua ou contrato específico.
    
    WIN$N -> WIN (MT5 usa série contínua sem sufixo)
    WING26 -> WING26 (mantém contrato específico)
    """
    # Remove sufixo $N se presente
    if "$N" in symbol:
        base = symbol.replace("$N", "")
        
        # MT5 usa nomenclatura diferente para série contínua
        # Algumas corretoras: WIN#, WIN!, WIN_continuous
        # Teste qual sua corretora usa:
        possible_names = [
            base,           # WIN
            f"{base}#",     # WIN#
            f"{base}!",     # WIN!
            f"{base}_C",    # WIN_C
        ]
        
        # Tenta cada nomenclatura
        for name in possible_names:
            if mt5.symbol_select(name, True):
                logger.info(f"✅ Série contínua encontrada: {name}")
                return name
        
        # Fallback: usa contrato mais próximo
        logger.warning(f"⚠️ Série contínua não encontrada. Buscando contrato atual...")
        
        # Busca contratos disponíveis com base no nome
        symbols = mt5.symbols_get(group=f"*{base}*")
        if symbols and len(symbols) > 0:
            # Ordena por volume (contrato mais líquido = mais próximo do vencimento)
            sorted_symbols = sorted(symbols, key=lambda s: s.volume, reverse=True)
            current_contract = sorted_symbols[0].name
            logger.info(f"📅 Usando contrato atual: {current_contract}")
            return current_contract
        
        # Se nada funcionar, retorna original
        logger.error(f"❌ Nenhum contrato encontrado para {symbol}")
        return symbol
    
    return symbol
```

**Use na função `load_futures_data_for_optimizer`** (linha ~165):

```python
# ANTES:
mt5.symbol_select(symbol, True)

# DEPOIS:
symbol_mt5 = normalize_futures_symbol(symbol)
mt5.symbol_select(symbol_mt5, True)
```

---

### ❌ PROBLEMA #2: PROFIT FACTOR 157.56 É IMPOSSÍVEL

**Análise:**
```
PF: 157.56
Win Rate: 22.2% (2 wins / 9 trades)
Trades: 9 (7 stops + 2 targets)
```

**Matemática:**
- Se PF = 157.56, isso significa: Lucro médio / Perda média = 157.56
- Com 22.2% WR: Você tem 2 trades vencedores e 7 perdedores
- **Para PF 157.56 acontecer**: Cada win precisa ser 550x maior que cada loss!

**Exemplo:**
- Loss médio: R$ 100
- Win médio: R$ 55.000 ❌ IRREAL

**O que está errado:**

1. **Bug no cálculo de PF** (optimizer_optuna.py, função compute_metrics)
2. **Overfitting extremo** (parâmetros se ajustaram a 2 trades específicos)
3. **Dados insuficientes** (WING26 tem poucos dias de histórico)

**CORREÇÃO:**

Arquivo: `optimizer_optuna.py` - Procure a função `compute_metrics`

Adicione validação:

```python
def compute_metrics(equity_curve: list, initial_capital: float = 100000.0) -> dict:
    """
    Calcula métricas com proteção contra valores irreais.
    """
    if not equity_curve or len(equity_curve) < 3:
        return {
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "sharpe": 0.0,
            "calmar": 0.0,
            "profit_factor": 0.0,
        }
    
    arr = np.array(equity_curve, dtype=float)
    
    # Total Return
    total_return = (arr[-1] - arr[0]) / arr[0]
    
    # Max Drawdown
    running_max = np.maximum.accumulate(arr)
    drawdown = (arr - running_max) / running_max
    max_dd = abs(np.min(drawdown)) if len(drawdown) > 0 else 0.0
    
    # Sharpe Ratio
    returns = np.diff(arr) / arr[:-1]
    if len(returns) > 5 and np.std(returns) > 0:
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 * 28)  # Anualizado
    else:
        sharpe = 0.0
    
    # Calmar Ratio
    calmar = (total_return / max_dd) if max_dd > 0 else 0.0
    
    # Profit Factor (CORRIGIDO)
    gross_profit = np.sum(returns[returns > 0]) if len(returns[returns > 0]) > 0 else 0.0
    gross_loss = abs(np.sum(returns[returns < 0])) if len(returns[returns < 0]) > 0 else 0.0
    
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
        
        # 🔧 VALIDAÇÃO CRÍTICA: PF > 10 é suspeito, > 50 é impossível
        if profit_factor > 10.0:
            logger.warning(f"⚠️ Profit Factor suspeito: {profit_factor:.2f} (capped to 10.0)")
            profit_factor = min(profit_factor, 10.0)  # Cap máximo
    else:
        profit_factor = 0.0 if gross_profit == 0 else 999.0  # Se só tem wins
    
    return {
        "total_return": float(total_return),
        "max_drawdown": float(max_dd),
        "sharpe": float(sharpe),
        "calmar": float(calmar),
        "profit_factor": float(profit_factor),
    }
```

---

### ❌ PROBLEMA #3: POUCOS TRADES (9 vs 15-30 esperado)

**Análise dos diagnósticos:**
```
c_setup: 17        → 17 setups identificados
c_volat: 25        → 25 BLOQUEADOS por volatilidade
c_ml: 0            → 0 bloqueados por ML (bom!)
c_vwap: 14         → 14 BLOQUEADOS por VWAP
c_success: 17      → Apenas 17 executados

Taxa de conversão: 17/17 = 100% (bom)
Taxa de bloqueio: (25+14)/56 = 69.6% ❌ MUITO ALTO
```

**Principais bloqueadores:**
1. **Volatilidade (25 bloqueios)** - Filtro ATR muito restritivo
2. **VWAP (14 bloqueios)** - Distância da VWAP muito pequena

**CORREÇÃO:**

Arquivo: `optimizer_optuna.py` - Linha ~816 (dentro de `fast_backtest_core`)

Procure por:

```python
# ❌ ANTES (linha ~220-240)
# Volatility filter
if atr[i] / price > MAX_ATR_PCT:
    continue  # Bloqueia se ATR% muito alto
```

**SUBSTITUA por:**

```python
# ✅ DEPOIS - Filtro adaptativo
# Calcula ATR% permitido baseado no ADX (força da tendência)
current_adx = adx[i]
current_atr_pct = atr[i] / max(price, 1e-6)

# Se ADX > 30 (tendência forte), permite ATR mais alto
if current_adx > 30:
    max_atr_allowed = 0.08  # 8% em tendência forte
elif current_adx > 20:
    max_atr_allowed = 0.06  # 6% em tendência média
else:
    max_atr_allowed = 0.04  # 4% em mercado lateral

# Só bloqueia se ATR excede o limite AJUSTADO
if current_atr_pct > max_atr_allowed:
    c_volat += 1  # Contador de bloqueios
    continue
```

**Para VWAP**, procure por (linha ~270):

```python
# ❌ ANTES
vwap_dist = abs(price - vwap[i]) / atr[i]
if vwap_dist > vwap_dist_thresh:  # Default: 1.5
    continue
```

**SUBSTITUA por:**

```python
# ✅ DEPOIS - Permite trades longe da VWAP se tendência forte
vwap_dist = abs(price - vwap[i]) / max(atr[i], 1e-6)

# Em tendências fortes (ADX > 25), permite distância maior da VWAP
if current_adx > 25:
    max_vwap_dist = vwap_dist_thresh * 2.0  # Dobra o limite
else:
    max_vwap_dist = vwap_dist_thresh

if vwap_dist > max_vwap_dist:
    c_vwap += 1
    continue
```

---

## 🎯 TESTE DE VALIDAÇÃO

Execute este comando para validar as correções:

```bash
# Teste 1: Verifica símbolo correto
python -c "
import MetaTrader5 as mt5
mt5.initialize()

# Testa diferentes nomenclaturas
symbols_to_test = ['WIN', 'WIN#', 'WIN!', 'WIN_C', 'WING26']
for sym in symbols_to_test:
    if mt5.symbol_select(sym, True):
        info = mt5.symbol_info(sym)
        print(f'✓ {sym}: Volume={info.volume}, Bid={info.bid}')
    else:
        print(f'✗ {sym}: Não encontrado')
"

# Teste 2: Otimização com diagnósticos expandidos
XP3_DEBUG=1 python otimizador_semanal.py --symbols WIN\$N --maxevals 50 --bars 5000

# Teste 3: Verifica PF no relatório
grep "Profit Factor" optimizer_output/weekly_*.txt
```

---

## 📋 CHECKLIST DE IMPLEMENTAÇÃO

Aplique as correções nesta ordem:

- [ ] **1. Normalização de símbolos** (normalize_futures_symbol)
- [ ] **2. Cap de Profit Factor** (compute_metrics)
- [ ] **3. Filtro ATR adaptativo** (fast_backtest_core)
- [ ] **4. Filtro VWAP flexível** (fast_backtest_core)
- [ ] **5. Re-teste completo** (50 trials, 5000 bars)

---

## 📊 RESULTADOS ESPERADOS APÓS CORREÇÕES

**ANTES:**
```
Símbolo: WING26
Trades: 9
PF: 157.56 (irreal)
Bloqueios: 69.6%
```

**DEPOIS:**
```
Símbolo: WIN (série contínua)
Trades: 25-45
PF: 1.5-3.5 (realista)
Bloqueios: 30-40%
```

---

## 🔬 MODO DEBUG AVANÇADO

Adicione ao início do seu teste:

```bash
# Exporta variáveis de debug
export XP3_DEBUG=1
export XP3_RELAX_VOLATILITY=1
export XP3_FORCE_LATERAL=0

# Executa com logging expandido
python otimizador_semanal.py \
  --symbols WIN\$N \
  --maxevals 100 \
  --bars 5000 \
  2>&1 | tee optimizer_debug.log
```

**Depois, analise:**
```bash
# Conta bloqueios
grep "c_volat" optimizer_debug.log | tail -20
grep "c_vwap" optimizer_debug.log | tail -20

# Verifica símbolo usado
grep "barras carregadas" optimizer_debug.log
```

---

## ⚡ QUICK FIX (Se não quiser alterar código agora)

Execute com parâmetros relaxados:

```bash
# Usa contrato específico (temporário)
python otimizador_semanal.py \
  --symbols WING26 \
  --maxevals 100 \
  --bars 2000 \
  --relax-volatility \
  --vwap-trend

# OU força série contínua (se corretora suporta)
python otimizador_semanal.py \
  --symbols WIN \
  --maxevals 100 \
  --bars 5000
```

---

## 📞 PRÓXIMOS PASSOS

1. **Implemente as 4 correções acima**
2. **Execute teste com 100 trials**
3. **Me envie o novo arquivo weekly_*.txt**
4. **Vamos analisar se PF está realista e trades aumentaram**

Se após isso ainda tiver < 20 trades, temos mais ajustes finos para fazer nos filtros de entrada.
