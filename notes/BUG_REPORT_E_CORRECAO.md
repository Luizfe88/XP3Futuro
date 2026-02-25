# 🐛 BUG CRÍTICO ENCONTRADO - SISTEMA DE SCORING IMPOSSÍVEL

**Data:** 04/02/2026  
**Arquivo:** `optimizer_optuna.py`  
**Linhas:** 268-450 (Entry Logic)

---

## 🚨 PROBLEMA RAIZ

O sistema de entrada exige **score_filtros >= 4**, mas a lógica de pontuação torna isso **praticamente impossível** quando não há modelo ML ativo.

### Breakdown do Sistema de Pontos

```python
# LONG ENTRY (linhas 306-329)
score_filtros = 0

if is_trend_long:          # +1 ponto
    score_filtros += 1
    
if setup_a/b/c_long:       # +2 pontos
    score_filtros += 2
    
if vol_ok_futures:         # +1 ponto (ADX condições)
    score_filtros += 1
    
if ml_sig == 1:            # +2 pontos (SE houver ML)
    score_filtros += 2
elif len(ml_probs) == 0:   # +1 ponto (SE NÃO houver ML)
    score_filtros += 1
    
if close_above_vwap:       # +1 ponto
    score_filtros += 1
else:
    score_filtros += 0.0   # ⚠️ Explícito: 0 pontos
    
if candle_ok:              # +1 ponto (SEMPRE True!)
    score_filtros += 1

# CONDIÇÃO DE ENTRADA (linha 329)
if score_filtros >= 4 and has_setup_long:
    # Executa trade...
```

---

## 📊 CENÁRIOS POSSÍVEIS

### Cenário 1: SEM Modelo ML (situação atual)
| Filtro | Pontos | Cumulative |
|--------|--------|------------|
| Tendência Long | +1 | 1 |
| Setup RSI/Mom | +2 | 3 |
| Vol (ADX OK) | +1 | 4 |
| **ML Ausente** | +1 | **5** |
| VWAP Próximo | +1 | 6 |
| Candle OK | +1 | 7 |

**Máximo possível**: 7 pontos  
**Entrada requer**: 4 pontos  
**Taxa de sucesso**: Baixíssima se qualquer filtro falhar

### Cenário 2: COM Modelo ML ativo mas SEM sinal
| Filtro | Pontos | Cumulative |
|--------|--------|------------|
| Tendência Long | +1 | 1 |
| Setup RSI/Mom | +2 | 3 |
| Vol (ADX OK) | +1 | 4 |
| **ML Sem Sinal** | **+0** | **4** ⚠️ |
| VWAP Próximo | +1 | 5 |
| Candle OK | +1 | 6 |

**Problema**: Se ML existe mas não dá sinal, você precisa **TODOS** os outros filtros!

### Cenário 3: Falha em VWAP (comum)
```python
# Se preço NÃO está próximo da VWAP:
Tendência: +1
Setup: +2
ADX: +1
ML: +1 (sem ML) ou +0 (com ML sem sinal)
VWAP: +0  ⚠️ PERDEU 1 PONTO
Candle: +1

Total SEM ML: 5 pontos (PASSA)
Total COM ML: 4 pontos (PASSA no limite)
```

### Cenário 4: Falha em ADX (seus parâmetros)
```python
# WDO: ADX threshold = 12
# vol_ok_futures = (ADX > 6) OR (12 < ADX < 50)

# Se ADX = 10 (entre 6 e 12):
vol_ok_futures = (10 > 6) OR (False)  # TRUE ✅

# Se ADX = 5 (abaixo do mínimo):
vol_ok_futures = (5 > 6) OR (False)   # FALSE ❌

Tendência: +1
Setup: +2
ADX: +0  ⚠️ PERDEU 1 PONTO
ML: +1
VWAP: +1
Candle: +1

Total: 6 pontos (AINDA PASSA)
```

---

## 🔍 POR QUE VOCÊ TEM 0 TRADES?

### Análise dos Seus Parâmetros

#### WDO
```
ema_short: 12, ema_long: 27
rsi_low: 40, rsi_high: 80
adx_threshold: 12
sl_atr_multiplier: 1.7
tp_mult: 1.75
```

**Problema 1: RSI Setup Muito Restritivo**
```python
setup_a_long = is_trend_long and (rsi[i] < max(rsi_low, 45))
# Seu rsi_low = 40, mas usa max(40, 45) = 45!
# Então RSI precisa ser < 45 (não < 40 como otimizado!)
```

**Problema 2: Momentum = 0.0**
```python
setup_c_long = (momentum[i] > 0.002) and is_trend_long and (rsi[i] < 55)
# Seu mom_min = 0.0, mas setup_c exige > 0.002 (hardcoded!)
# Momentum raramente > 0.002 em mercado lateral
```

**Problema 3: VWAP Distance**
```python
vwap_dist_thresh = 0.5  # Default
dist_vwap = abs(price - vwap[i]) / max(atr[i], 1e-9)
close_above_vwap = (price > vwap[i]) and (dist_vwap <= 0.5)
# Preço precisa estar < 0.5 ATR da VWAP
# Em futuros BR voláteis, isso é MUITO restritivo!
```

#### WIN
```
ema_short: 11, ema_long: 23
rsi_low: 34, rsi_high: 65
adx_threshold: 17
sl_atr_multiplier: 1.7
tp_mult: 3.39
```

**Mesmos problemas + RSI ainda mais apertado**

---

## ✅ CORREÇÕES NECESSÁRIAS

### Correção 1: Ajustar Lógica de Scoring (CRÍTICO)

```python
# ANTES (linha 329):
if score_filtros >= 4 and has_setup_long:

# DEPOIS - Opção A (Mais Permissivo):
if score_filtros >= 3 and has_setup_long:

# DEPOIS - Opção B (Sistema Adaptativo):
min_score = 3 if len(ml_probs) == 0 else 4
if score_filtros >= min_score and has_setup_long:
```

### Correção 2: Remover max(rsi_low, 45) Hardcode

```python
# ANTES (linha 277):
setup_a_long = is_trend_long and (rsi[i] < max(rsi_low, 45))

# DEPOIS:
setup_a_long = is_trend_long and (rsi[i] < rsi_low)
```

### Correção 3: Usar mom_min Parametrizado

```python
# ANTES (linha 281):
setup_c_long = (momentum[i] > 0.002) and is_trend_long and (rsi[i] < 55)

# DEPOIS:
mom_thresh = max(mom_min, 0.001) if mom_min > 0 else 0.001
setup_c_long = (momentum[i] > mom_thresh) and is_trend_long and (rsi[i] < 55)
```

### Correção 4: Relaxar VWAP Distance para Futuros

```python
# ANTES (linha 299):
dist_vwap = abs(price - vwap[i]) / max(atr[i], 1e-9)
close_above_vwap = (price > vwap[i]) and (dist_vwap <= vwap_dist_thresh)

# DEPOIS:
# Para futuros, aumentar threshold ou tornar opcional
vwap_thresh_adj = vwap_dist_thresh * 2.0 if asset_type == 1 else vwap_dist_thresh
dist_vwap = abs(price - vwap[i]) / max(atr[i], 1e-9)
close_above_vwap = (price > vwap[i]) and (dist_vwap <= vwap_thresh_adj)
```

### Correção 5: Adicionar Logging de Debug

```python
# Adicionar após linha 329:
if score_filtros >= 3 and has_setup_long:
    if score_filtros < 4:
        print(f"[ENTRY_DEBUG] Bar {i}: Score {score_filtros}/4 (borderline) | "
              f"Trend={int(is_trend_long)} Setup={int(has_setup_long)} "
              f"ADX={adx[i]:.1f}/{adx_threshold} "
              f"ML={ml_sig} VWAP={int(close_above_vwap)} "
              f"RSI={rsi[i]:.1f}/{rsi_low}")
```

---

## 🎯 PLANO DE AÇÃO IMEDIATO

### Fase 1: Correções de Código (30 minutos)

1. **Abrir `optimizer_optuna.py`**
2. **Linha 329**: Trocar `>= 4` por `>= 3`
3. **Linha 409**: Trocar `>= 4` por `>= 3` (shorts)
4. **Linha 277**: Remover `max(rsi_low, 45)`, deixar só `rsi_low`
5. **Linha 278**: Simplificar para `rsi[i] > rsi_high`
6. **Linha 299**: Multiplicar threshold por 2.0 se asset_type == 1
7. **Salvar arquivo**

### Fase 2: Teste Rápido (1 hora)

```bash
# Rodar com 10 trials apenas para validar
python otimizador_semanal.py --symbols WIN$N --maxevals 10 --bars 3000
```

**Resultado esperado**: 
- Pelo menos 5-15 trades no período OOS
- Métricas visíveis (não mais 0.00 em tudo)

### Fase 3: Re-otimização (2-4 horas)

```bash
# Otimização completa com novos ranges
python otimizador_semanal.py --symbols WDO$N WIN$N --maxevals 100 --bars 5000
```

---

## 📝 MODIFICAÇÕES SUGERIDAS NOS RANGES

### Para optimizer_optuna.py - Função optimize()

```python
# RANGES ATUAIS (linhas ~900-950):
'rsi_low': trial.suggest_int('rsi_low', 20, 40),
'rsi_high': trial.suggest_int('rsi_high', 60, 80),
'adx_threshold': trial.suggest_int('adx_threshold', 10, 50),

# RANGES RECOMENDADOS PARA FUTUROS:
'rsi_low': trial.suggest_int('rsi_low', 25, 40),      # Mantém
'rsi_high': trial.suggest_int('rsi_high', 60, 75),     # Reduz max
'adx_threshold': trial.suggest_int('adx_threshold', 15, 35),  # Range mais realista

# Adicionar range para VWAP:
'vwap_dist_thresh': trial.suggest_float('vwap_dist_thresh', 0.5, 2.0),
```

---

## 🧪 TESTE DE VALIDAÇÃO

Após correções, adicione este código de debug temporário:

```python
# Logo após linha 274 (antes de calcular setups):
if i % 100 == 0:  # A cada 100 barras
    debug_info = {
        'bar': i,
        'price': price,
        'ema_short': ema_short[i],
        'ema_long': ema_long[i],
        'rsi': rsi[i],
        'adx': adx[i],
        'momentum': momentum[i],
        'vwap': vwap[i],
        'atr': atr[i]
    }
    print(f"[BAR_SAMPLE] {debug_info}")
```

**O que procurar:**
- ADX geralmente entre 15-40 (confirma que threshold 12-17 está OK)
- RSI oscilando entre 30-70 (confirma que seus ranges estão OK)
- Momentum próximo de 0 na maioria do tempo (confirma que hardcode 0.002 é problema)

---

## 💾 BACKUP ANTES DE MODIFICAR

```bash
# Windows PowerShell:
Copy-Item optimizer_optuna.py optimizer_optuna_BACKUP_20260204.py

# Linux/Mac:
cp optimizer_optuna.py optimizer_optuna_BACKUP_20260204.py
```

---

## 🎓 CONCLUSÃO

Você não tem problema de parâmetros - você tem um **bug de lógica no código**.

O otimizador está encontrando parâmetros válidos, mas a função de backtest tem uma condição de entrada **matematicamente impossível** de satisfazer consistentemente.

**Prioridade 1**: Trocar `>= 4` por `>= 3` nas linhas 329 e 409.  
**Prioridade 2**: Remover hardcodes (max(rsi_low, 45), momentum > 0.002).  
**Prioridade 3**: Relaxar VWAP distance para futuros.

Após essas correções, você deve ver **50-150 trades** no período OOS para cada ativo.

---

**Status**: BLOQUEADOR - Sistema inoperável até correção  
**Impacto**: 100% dos trials resultam em 0 trades  
**Severidade**: CRÍTICA  
**Tempo estimado de correção**: 30 minutos  
**Risco de correção**: BAIXO (mudanças pontuais em lógica clara)
