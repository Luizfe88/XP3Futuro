# 🔬 DIAGNÓSTICO DETALHADO - ZERO TRADES APÓS CORREÇÕES

**Data:** 04/02/2026  
**Status:** Correções implementadas mas ainda 0 trades  
**Investigação:** Análise profunda linha por linha

---

## ✅ CONFIRMADO: Correções Foram Aplicadas

Verifiquei o código enviado:

1. ✅ **Linha 322:** `if score_filtros >= 4 and has_setup_long:` (CORRETO)
2. ✅ **Linha 402:** `if score_filtros_short >= 4:` (CORRETO)
3. ✅ **Linhas 414-456:** Código de entrada SHORT movido para dentro do `if` (CORRETO)

**Conclusão:** As correções estão OK. O problema está em **OUTRO LUGAR**.

---

## 🎯 HIPÓTESES INVESTIGADAS

### Hipótese #1: ML Desabilitado ❌

**Código (Linha 284-289):**
```python
ml_sig = 0
if len(ml_probs) > 0 and i < len(ml_probs):
    p = ml_probs[i]
    if p > ml_threshold:  # 0.54
        ml_sig = 1
```

**Código (Linha 716-723):**
```python
disable_ml = os.getenv("XP3_DISABLE_ML", "0") == "1"
if disable_ml or ml_model is None:
    ml_probs = np.array([], dtype=np.float64)  # ❌ Array vazio!
```

**Problema identificado:**
- `ml_probs` está VAZIO
- `ml_sig` fica em 0
- **PERDE 2 PONTOS** no score

**Linha 309-313 (Compensação):**
```python
if ml_sig == 1:
    c_ml += 1
    score_filtros += 2
elif len(ml_probs) == 0:  # ✅ Tem compensação
    score_filtros += 1
```

**Resultado:** Ganha 1 ponto ao invés de 2. Score máximo SEM ML = 7 pontos.

---

### Hipótese #2: VWAP Muito Restritivo ⚠️

**Código (Linha 292-294):**
```python
dist_vwap = abs(price - vwap[i]) / max(atr[i], 1e-9)
close_above_vwap = (price > vwap[i]) and (dist_vwap <= vwap_dist_thresh)  # 0.5
```

**Parâmetro:** `vwap_dist_thresh = 0.5` ATR

**Problema:**
- Em futuros voláteis, exigir distância < 0.5 ATR do VWAP é MUITO restritivo
- Se não passar: **PERDE 1 PONTO**

**Linha 314-318:**
```python
if close_above_vwap:
    c_vwap += 1
    score_filtros += 1
else:
    score_filtros += 0.0  # ❌ Zero pontos
```

---

### Hipótese #3: Volatilidade Multiplier 🔍

**Código (Linha 296):**
```python
vol_ok_futures = (adx[i] > adx_threshold * volatility_multiplier) or ((adx[i] > adx_threshold) and (adx[i] < 50))
```

**Parâmetro:** `volatility_multiplier = 0.7` (linha 778)

**Cálculo:**
- DOLH26: ADX threshold = 18 → 18 * 0.7 = **12.6**
- WING26: ADX threshold = 20 → 20 * 0.7 = **14.0**

**Problema:** Thresholds muito BAIXOS podem permitir mercado lateral demais.

Mas a condição tem um OR com `adx > threshold AND adx < 50`, então se ADX estiver entre 18-50, passa.

---

### Hipótese #4: Condição de Setup RSI ⚠️

**Código (Linha 277-282):**
```python
setup_a_long = is_trend_long and (rsi[i] < rsi_low)
setup_a_short = is_trend_short and (rsi[i] > rsi_high)
setup_b_long = (rsi_2[i] < 20)
setup_b_short = (rsi_2[i] > 80)
has_setup_long = setup_a_long or setup_b_long
has_setup_short = (setup_a_short or setup_b_short) and enable_shorts
```

**Parâmetros DOLH26:**
- `rsi_low = 36` → RSI < 36 (oversold)
- `rsi_high = 78` → RSI > 78 (overbought)

**Problema:**
- RSI raramente fica > 78 (extremo overbought)
- RSI < 36 é um pouco mais comum, mas ainda restritivo
- RSI_2 < 20 ou > 80 é MUITO RARO

**Probabilidade de setup:**
- `setup_a_long`: Baixa (RSI < 36 em tendência de alta)
- `setup_b_long`: Muito baixa (RSI_2 < 20 é extremo)
- **Resultado:** Poucos setups detectados

---

### Hipótese #5: Verificação de Tendência (CRÍTICO!) 🚨

**Código (Linha 725-728):**
```python
trend_freq = np.sum(ema_s > ema_l) / len(close)
if trend_freq < 0.30:
    logger.warning(f"[WARN] {symbol}: Mercado sem tendência clara (Alta em apenas {trend_freq:.1%})")
```

**Atenção:** Este código só da WARNING, não bloqueia!

---

## 🔍 ANÁLISE DE SCORE POR COMPONENTE

### Score Máximo Possível (sem ML):

| Componente | Pontos | Condição |
|------------|--------|----------|
| Trend (EMA) | 1 | `ema_short > ema_long` |
| Setup (RSI) | 2 | `rsi < 36` OU `rsi_2 < 20` |
| Volatilidade (ADX) | 1 | `adx > 18` (DOLH26) |
| ML (compensação) | 1 | `len(ml_probs) == 0` |
| VWAP | 1 | `dist < 0.5 ATR` |
| Candle | 1 | Sempre True |
| **TOTAL** | **7** | |

**Score mínimo necessário:** 4  
**Taxa de aprovação:** 4/7 = 57%

---

## 🎯 CENÁRIO PROBLEMA MAIS PROVÁVEL

### Cenário 1: Setup RSI não acontece
```
✅ Trend: 1 ponto (EMA 10 > EMA 20)
❌ Setup: 0 pontos (RSI não está < 36, nem RSI_2 < 20)
TOTAL: 1 ponto → NÃO ENTRA (precisa de 4)
```

Se **não há setup**, a condição na linha 290 falha:
```python
if has_setup_long or has_setup_short:
    # Todo o código de score está AQUI DENTRO
```

**🚨 PROBLEMA IDENTIFICADO:**

Se `has_setup_long = False` e `has_setup_short = False`, o código **NUNCA ENTRA** no bloco de cálculo de score!

---

## 🔬 TESTE DE VALIDAÇÃO

### O que verificar nos logs:

1. **Setups identificados** (linha 789):
```python
print(f"[DEBUG] [{symbol}] Funnel: Setups={int(total_setups)} | ...")
```

Se `Setups = 0`, o problema é que **nenhum setup RSI está sendo detectado**.

---

## ✅ SOLUÇÕES PROPOSTAS

### SOLUÇÃO #1: Relaxar Condições de Setup RSI (CRÍTICA)

**Problema:** RSI < 36 e RSI > 78 são muito extremos.

**Correção:** Ampliar ranges ou adicionar setup alternativo.

**Opção A - Ampliar RSI:**
```python
# Linha 760-761
params.get("rsi_low", 30),   # ANTES: 30 (agora vem 36 do Optuna)
params.get("rsi_high", 70),  # ANTES: 70 (agora vem 78 do Optuna)
```

**Problema:** Os parâmetros vêm do Optuna (36/78), então precisamos mudar o Optuna.

**Opção B - Adicionar Setup Alternativo (Momentum):**

Adicione após linha 279:
```python
setup_b_long = (rsi_2[i] < 20)
setup_b_short = (rsi_2[i] > 80)

# ✅ ADICIONAR SETUP C (Momentum)
setup_c_long = (momentum[i] > 0.001) and is_trend_long
setup_c_short = (momentum[i] < -0.001) and is_trend_short

has_setup_long = setup_a_long or setup_b_long or setup_c_long
has_setup_short = (setup_a_short or setup_b_short or setup_c_short) and enable_shorts
```

**Opção C - Forçar RSI mais flexível:**

Linha 277:
```python
# ANTES
setup_a_long = is_trend_long and (rsi[i] < rsi_low)

# DEPOIS (mais flexível)
setup_a_long = is_trend_long and (rsi[i] < max(rsi_low, 45))  # Pelo menos RSI < 45
```

---

### SOLUÇÃO #2: Relaxar VWAP Distance

**Linha 777:**
```python
# ANTES
float(params.get("vwap_dist_thresh", 0.5)),

# DEPOIS
float(params.get("vwap_dist_thresh", 1.5)),  # 1.5 ATR ao invés de 0.5
```

---

### SOLUÇÃO #3: Forçar ML em Modo Diagnóstico

**Ativar ML forçado:**
```bash
export XP3_FORCE_ML_DIAG=1
```

Isso cria um array ML com probabilidade 0.85, garantindo 2 pontos.

---

### SOLUÇÃO #4: Reduzir Score Ainda Mais (Temporário)

**Teste diagnóstico:**

Linha 322 e 402:
```python
# TESTE: Reduzir para 3 temporariamente
if score_filtros >= 3 and has_setup_long:
```

**ATENÇÃO:** Isso é apenas para DIAGNOSTICAR. Não deixe em produção.

---

## 📊 PLANO DE AÇÃO RECOMENDADO

### Passo 1: Adicionar Logging Detalhado

Adicione após linha 290:
```python
if has_setup_long or has_setup_short:
    c_setup += 1
    # ✅ ADICIONAR LOG
    if (i % 100 == 0):  # A cada 100 barras
        print(f"[DIAG] i={i} | setup_long={has_setup_long} setup_short={has_setup_short} | "
              f"rsi={rsi[i]:.1f} rsi_low={rsi_low} rsi_high={rsi_high} | "
              f"rsi_2={rsi_2[i]:.1f} | trend_long={is_trend_long}")
```

### Passo 2: Executar e Verificar Logs

```bash
python otimizador_semanal.py 2>&1 | grep -E "(DIAG|Funnel)"
```

### Passo 3: Implementar Correção Baseada nos Logs

**Se Setups = 0:**
- Implementar SOLUÇÃO #1 (adicionar setup alternativo)

**Se Setups > 0 mas Executed = 0:**
- Implementar SOLUÇÃO #2 (relaxar VWAP)
- Ou SOLUÇÃO #3 (forçar ML)

---

## 🎓 CONCLUSÃO TÉCNICA

O problema mais provável é uma **combinação de fatores**:

1. ✅ **Correções aplicadas** (score >= 4)
2. ❌ **Setup RSI muito restritivo** (36/78 + RSI_2 20/80)
3. ❌ **VWAP distance muito apertado** (0.5 ATR)
4. ⚠️ **ML desabilitado** (perde 1 ponto)

**Cenário provável:**
- Poucas barras atendem `rsi < 36` OU `rsi_2 < 20`
- `has_setup_long = False` na maioria do tempo
- Código nunca entra no bloco de cálculo de score
- **Resultado:** 0 trades

**Solução prioritária:**

1. **Adicionar setup alternativo (momentum)** - IMPLEMENTAR AGORA
2. **Relaxar VWAP para 1.5 ATR** - IMPLEMENTAR AGORA
3. **Forçar ML diagnóstico** - TESTAR
4. **Adicionar logging detalhado** - DIAGNOSTICAR

---

## 📝 CÓDIGO COMPLETO DA CORREÇÃO

### Correção #1: Setup Alternativo (LINHA 279)

```python
# ANTES
setup_b_long = (rsi_2[i] < 20)
setup_b_short = (rsi_2[i] > 80)
has_setup_long = setup_a_long or setup_b_long
has_setup_short = (setup_a_short or setup_b_short) and enable_shorts

# DEPOIS
setup_b_long = (rsi_2[i] < 20)
setup_b_short = (rsi_2[i] > 80)

# ✅ Setup C: Momentum em tendência (mais comum)
setup_c_long = (momentum[i] > 0.002) and is_trend_long and (rsi[i] < 55)
setup_c_short = (momentum[i] < -0.002) and is_trend_short and (rsi[i] > 45)

has_setup_long = setup_a_long or setup_b_long or setup_c_long
has_setup_short = (setup_a_short or setup_b_short or setup_c_short) and enable_shorts
```

### Correção #2: VWAP Threshold (LINHA 777)

```python
# ANTES
float(params.get("vwap_dist_thresh", 0.5)),

# DEPOIS
float(params.get("vwap_dist_thresh", 1.5)),
```

### Correção #3: Logging Diagnóstico (LINHA 291)

```python
if has_setup_long or has_setup_short:
    c_setup += 1
    
    # ✅ Log diagnóstico
    if (i % 50 == 0):
        print(f"[DIAG] Bar {i}: setup_L={int(has_setup_long)} setup_S={int(has_setup_short)} | "
              f"RSI={rsi[i]:.1f}({rsi_low}/{rsi_high}) RSI2={rsi_2[i]:.1f} MOM={momentum[i]:.4f} | "
              f"trend_L={int(is_trend_long)} ADX={adx[i]:.1f}")
```

---

**Após estas correções, você deve ver trades sendo executadas!**

---

**Desenvolvido por:** Especialista em Trading Algorítmico  
**Última Atualização:** 04/02/2026 11:00
