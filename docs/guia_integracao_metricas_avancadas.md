# 🎯 GUIA DE INTEGRAÇÃO - MÉTRICAS AVANÇADAS PARA FUTUROS

## 📋 Resumo Executivo

Este guia mostra como integrar as **métricas profissionais** no seu otimizador de futuros.

### ✅ O que foi implementado:

1. **Recovery Factor** - Velocidade de recuperação (melhor que Sharpe)
2. **Expectancy** - R$ ganhos por trade (crítico para custos)
3. **Sortino Ratio** - Penaliza só volatilidade negativa
4. **SQN** - System Quality Number (Van Tharp)
5. **MAE/MFE** - Ajuste perfeito de SL/TP
6. **Ulcer Index** - "Dor" do drawdown
7. **PF Ajustado** - Profit Factor após custos B3

### 🔥 VALIDAÇÃO OBRIGATÓRIA: Mínimo 20 trades

---

## 🚀 Passo 1: Adicionar o Módulo

Copie o arquivo `advanced_metrics_futures.py` para a pasta do projeto:

```bash
cp advanced_metrics_futures.py c:/Users/luizf/Documents/xp3future/
```

---

## 🔧 Passo 2: Modificar otimizador_semanal.py

### 2.1 Adicionar Import

**Localizar linha ~100 (após imports) e adicionar:**

```python
# ============================================================================
# MÉTRICAS AVANÇADAS PARA FUTUROS
# ============================================================================
try:
    from advanced_metrics_futures import (
        calculate_all_advanced_metrics,
        format_metrics_report,
        MIN_TRADES_REQUIRED
    )
    ADVANCED_METRICS_ENABLED = True
    logger.info("✅ Métricas avançadas para futuros carregadas")
except ImportError:
    ADVANCED_METRICS_ENABLED = False
    MIN_TRADES_REQUIRED = 20  # Fallback
    logger.warning("⚠️ advanced_metrics_futures não encontrado - usando métricas básicas")
```

---

### 2.2 Modificar Função `backtest_params_on_df`

**Localizar a função que retorna as métricas do backtest e modificar o retorno:**

**ANTES (exemplo genérico):**
```python
def backtest_params_on_df(symbol, params, df, ml_model=None):
    # ... código do backtest ...
    
    return {
        "total_trades": len(trades),
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "total_return": total_return,
        "equity_curve": equity_curve
    }
```

**DEPOIS:**
```python
def backtest_params_on_df(symbol, params, df, ml_model=None):
    # ... código do backtest ...
    
    # Métricas básicas (manter compatibilidade)
    basic_metrics = {
        "total_trades": len(trades),
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "total_return": total_return,
        "equity_curve": equity_curve
    }
    
    # ═══════════════════════════════════════════════════════════════
    # 🔥 MÉTRICAS AVANÇADAS (SE HABILITADO)
    # ═══════════════════════════════════════════════════════════════
    if ADVANCED_METRICS_ENABLED and len(trades) > 0:
        try:
            # Calcula todas as métricas avançadas
            adv_metrics = calculate_all_advanced_metrics(
                trades=trades,
                equity_curve=equity_curve,
                total_pnl=total_return * initial_capital,
                initial_capital=initial_capital,
                cost_per_trade=28.0,  # WIN: corretagem + B3 + slippage
                risk_free_rate=0.11   # Selic ~11%
            )
            
            # Adiciona ao retorno
            basic_metrics['advanced'] = adv_metrics
            
            # Log se sistema é válido
            if not adv_metrics.is_valid:
                logger.warning(
                    f"⚠️ {symbol}: {adv_metrics.validation_message}"
                )
            else:
                logger.info(
                    f"✅ {symbol}: SQN={adv_metrics.sqn:.2f} | "
                    f"RF={adv_metrics.recovery_factor:.2f} | "
                    f"Exp=R${adv_metrics.expectancy:.2f}"
                )
        
        except Exception as e:
            logger.error(f"Erro ao calcular métricas avançadas para {symbol}: {e}")
            basic_metrics['advanced'] = None
    
    return basic_metrics
```

---

### 2.3 Modificar Critério de Seleção (CRÍTICO!)

**Localizar onde os resultados são filtrados/ranqueados (~linha 2310+):**

**ANTES:**
```python
# Ordena por Calmar ou Profit Factor
opp_sorted = sorted(
    opp_candidates,
    key=lambda x: float(x.get("res", {}).get("test_metrics", {}).get("calmar", 0.0)),
    reverse=True
)
```

**DEPOIS:**
```python
# ═══════════════════════════════════════════════════════════════
# 🎯 SELEÇÃO COM MÉTRICAS AVANÇADAS
# ═══════════════════════════════════════════════════════════════

def calculate_selection_score(result):
    """
    Score de seleção usando métricas avançadas.
    
    Prioridade:
    1. Sistema VÁLIDO (>= 20 trades)
    2. SQN (30%)
    3. Recovery Factor (25%)
    4. Sortino (20%)
    5. Expectancy (15%)
    6. PF Ajustado (10%)
    """
    metrics = result.get("res", {}).get("test_metrics", {})
    adv = metrics.get("advanced")
    
    # Se não tem métricas avançadas ou sistema inválido, usa score baixo
    if not adv or not adv.is_valid:
        return 0.0
    
    # Usa o score final já calculado
    return adv.final_score

# Ordena por score avançado
opp_sorted = sorted(
    opp_candidates,
    key=calculate_selection_score,
    reverse=True
)

# Log dos top 5
logger.info("\n🏆 TOP 5 SISTEMAS (por métricas avançadas):")
for i, item in enumerate(opp_sorted[:5], 1):
    sym = item['symbol']
    adv = item.get("res", {}).get("test_metrics", {}).get("advanced")
    
    if adv and adv.is_valid:
        logger.info(
            f"{i}. {sym}: "
            f"Score={adv.final_score:.1f} ({adv.grade}) | "
            f"SQN={adv.sqn:.2f} ({adv.sqn_classification}) | "
            f"RF={adv.recovery_factor:.2f} ({adv.recovery_classification}) | "
            f"Exp=R${adv.expectancy:.2f}"
        )
    else:
        logger.warning(f"{i}. {sym}: Métricas inválidas (<20 trades)")
```

---

### 2.4 Adicionar Filtro de Validação

**ANTES de selecionar os elite, adicionar:**

```python
# ═══════════════════════════════════════════════════════════════
# 🔒 FILTRO DE VALIDAÇÃO: Remove sistemas com < 20 trades
# ═══════════════════════════════════════════════════════════════

if ADVANCED_METRICS_ENABLED:
    valid_systems = []
    rejected_low_trades = []
    
    for item in opp_sorted:
        sym = item['symbol']
        adv = item.get("res", {}).get("test_metrics", {}).get("advanced")
        
        if adv and adv.is_valid:
            valid_systems.append(item)
        else:
            n_trades = item.get("res", {}).get("test_metrics", {}).get("total_trades", 0)
            rejected_low_trades.append((sym, n_trades))
    
    if rejected_low_trades:
        logger.warning(f"\n⚠️ {len(rejected_low_trades)} sistemas REJEITADOS (< {MIN_TRADES_REQUIRED} trades):")
        for sym, n in rejected_low_trades[:10]:  # Mostra até 10
            logger.warning(f"   - {sym}: {n} trades")
    
    # Usa apenas sistemas válidos
    opp_sorted = valid_systems
    logger.info(f"✅ {len(valid_systems)} sistemas VÁLIDOS para seleção final")
```

---

### 2.5 Melhorar Relatório Final

**Localizar onde salva o relatório (~linha 2560+) e adicionar:**

```python
# ═══════════════════════════════════════════════════════════════
# 📊 RELATÓRIO DETALHADO COM MÉTRICAS AVANÇADAS
# ═══════════════════════════════════════════════════════════════

try:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    advanced_report_path = os.path.join(OPT_OUTPUT_DIR, f"advanced_metrics_{ts}.md")
    
    with open(advanced_report_path, "w", encoding="utf-8") as f:
        f.write("# 📊 RELATÓRIO AVANÇADO - MÉTRICAS PROFISSIONAIS PARA FUTUROS\n\n")
        f.write(f"**Gerado em:** {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n\n")
        f.write("---\n\n")
        
        for sym in sorted(final_elite.keys()):
            res = final_elite[sym]
            metrics = res.get("test_metrics", {})
            adv = metrics.get("advanced")
            
            f.write(f"## 🎯 {sym}\n\n")
            
            if adv and adv.is_valid:
                # Usa a função de formatação profissional
                report_text = format_metrics_report(adv)
                f.write(report_text)
                f.write("\n\n")
            else:
                f.write(f"### ⚠️ Sistema Inválido\n")
                f.write(f"Trades: {metrics.get('total_trades', 0)} (< {MIN_TRADES_REQUIRED} mínimos)\n\n")
            
            f.write("---\n\n")
    
    logger.info(f"📄 Relatório avançado salvo: {advanced_report_path}")

except Exception as e:
    logger.error(f"Erro ao gerar relatório avançado: {e}")
```

---

## 📊 Passo 3: Modificar optimizer_optuna.py (se usado)

**Se você usa Optuna para otimização, modificar a função objetivo:**

```python
def objective(trial):
    # ... define parâmetros ...
    
    # Executa backtest
    result = backtest_params_on_df(symbol, params, df_train)
    
    # ═══════════════════════════════════════════════════════════════
    # 🎯 OTIMIZAÇÃO POR MÉTRICAS AVANÇADAS
    # ═══════════════════════════════════════════════════════════════
    
    adv = result.get('advanced')
    
    # REJEITA se < 20 trades
    if not adv or not adv.is_valid:
        return -999.0  # Score muito baixo
    
    # Score combinado (customizável)
    score = (
        adv.sqn * 0.40 +              # 40%: Qualidade do sistema
        adv.recovery_factor * 0.30 +  # 30%: Velocidade recuperação
        adv.sortino_ratio * 0.20 +    # 20%: Controle de risco
        adv.expectancy * 0.10         # 10%: Expectativa
    )
    
    return score
```

---

## 🎨 Passo 4: Visualização (Opcional mas Recomendado)

**Criar gráfico comparativo:**

```python
import matplotlib.pyplot as plt

def plot_metrics_comparison(systems_metrics):
    """
    Plota comparação visual das métricas avançadas.
    
    Args:
        systems_metrics: Dict {symbol: AdvancedMetrics}
    """
    symbols = list(systems_metrics.keys())
    
    # Prepara dados
    sqn_values = [m.sqn for m in systems_metrics.values()]
    rf_values = [m.recovery_factor for m in systems_metrics.values()]
    sortino_values = [m.sortino_ratio for m in systems_metrics.values()]
    exp_values = [m.expectancy for m in systems_metrics.values()]
    
    # Cria subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('📊 Comparação de Métricas Avançadas', fontsize=16, fontweight='bold')
    
    # 1. SQN
    axes[0, 0].barh(symbols, sqn_values, color='skyblue')
    axes[0, 0].axvline(x=2.5, color='orange', linestyle='--', label='BOM (2.5)')
    axes[0, 0].axvline(x=3.0, color='green', linestyle='--', label='MUITO BOM (3.0)')
    axes[0, 0].set_xlabel('SQN (Van Tharp)')
    axes[0, 0].set_title('System Quality Number')
    axes[0, 0].legend()
    
    # 2. Recovery Factor
    axes[0, 1].barh(symbols, rf_values, color='lightgreen')
    axes[0, 1].axvline(x=3.0, color='orange', linestyle='--', label='BOM (3.0)')
    axes[0, 1].axvline(x=5.0, color='green', linestyle='--', label='EXCELENTE (5.0)')
    axes[0, 1].set_xlabel('Recovery Factor')
    axes[0, 1].set_title('Velocidade de Recuperação')
    axes[0, 1].legend()
    
    # 3. Sortino
    axes[1, 0].barh(symbols, sortino_values, color='salmon')
    axes[1, 0].axvline(x=1.0, color='orange', linestyle='--', label='ACEITÁVEL (1.0)')
    axes[1, 0].axvline(x=2.0, color='green', linestyle='--', label='BOM (2.0)')
    axes[1, 0].set_xlabel('Sortino Ratio')
    axes[1, 0].set_title('Controle de Volatilidade Negativa')
    axes[1, 0].legend()
    
    # 4. Expectancy
    axes[1, 1].barh(symbols, exp_values, color='gold')
    axes[1, 1].axvline(x=50, color='red', linestyle='--', label='Mínimo (R$50)')
    axes[1, 1].axvline(x=100, color='green', linestyle='--', label='Bom (R$100)')
    axes[1, 1].set_xlabel('Expectancy (R$)')
    axes[1, 1].set_title('R$ Ganho por Trade')
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    # Salva
    plt.savefig(os.path.join(OPT_OUTPUT_DIR, 'advanced_metrics_comparison.png'), dpi=300)
    logger.info("📈 Gráfico de métricas salvo: advanced_metrics_comparison.png")
```

---

## ✅ Checklist de Integração

- [ ] `advanced_metrics_futures.py` copiado para projeto
- [ ] Import adicionado em `otimizador_semanal.py`
- [ ] Função `backtest_params_on_df` modificada
- [ ] Critério de seleção atualizado (usa `final_score`)
- [ ] Filtro de validação (mín 20 trades) implementado
- [ ] Relatório avançado adicionado
- [ ] `optimizer_optuna.py` atualizado (se usado)
- [ ] Testado com um contrato (ex: WING25)
- [ ] Verificado logs de validação

---

## 🎯 Resultado Esperado

Após integração, você verá:

```
✅ Métricas avançadas para futuros carregadas
📊 DETECTANDO CONTRATOS VIGENTES...
✅ Contratos para otimizar: ['WING25', 'WDOG25', 'INDK25']

🔍 VALIDANDO LIQUIDEZ...
  ✅ WING25: OI=285,432 (mín: 200,000)
  ✅ WDOG25: OI=152,891 (mín: 100,000)

⚙️ INICIANDO OTIMIZAÇÃO DE 2 CONTRATOS...
✅ WING25: SQN=2.87 | RF=4.52 | Exp=R$78.45
✅ WDOG25: SQN=2.34 | RF=3.21 | Exp=R$62.30

🏆 TOP 5 SISTEMAS (por métricas avançadas):
1. WING25: Score=82.3 (B+) | SQN=2.87 (BOM) | RF=4.52 (BOM) | Exp=R$78.45
2. WDOG25: Score=75.8 (B) | SQN=2.34 (MÉDIO) | RF=3.21 (BOM) | Exp=R$62.30

⚠️ 3 sistemas REJEITADOS (< 20 trades):
   - INDK25: 12 trades
   - WSPG25: 8 trades

✅ 2 sistemas VÁLIDOS para seleção final
📄 Relatório avançado salvo: advanced_metrics_20250129_235959.md
```

---

## 📚 Referências

- **Van Tharp**: "Trade Your Way to Financial Freedom" (SQN)
- **John Sweeney**: "Maximum Adverse Excursion" (MAE/MFE)
- **Sortino & van der Meer**: "Downside Risk" (Sortino Ratio)
- **Jack Schwager**: "Market Wizards" (Recovery Factor)

---

## 💡 Dicas Finais

1. **Priorize SQN**: É a métrica mais robusta para validar se o sistema é estatístico
2. **Expectancy crítico**: Se < R$ 50 no WIN, revise a estratégia (custos vão comer lucro)
3. **Recovery Factor**: Mais importante que Sharpe para futuros (alavancados)
4. **MAE/MFE**: Use para ajustar SL/TP de forma científica (não "feeling")
5. **Mínimo 20 trades**: NÃO NEGOCIÁVEL - menos que isso é "sorte"

---

## 🚨 Troubleshooting

**Problema**: "ImportError: advanced_metrics_futures not found"
```bash
# Solução: Verifique se o arquivo está no path correto
ls -la advanced_metrics_futures.py
# Ou adicione ao PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/caminho/para/xp3future"
```

**Problema**: "Todos sistemas rejeitados (< 20 trades)"
```python
# Solução 1: Aumentar período de backtest
bt_config["BARS"] = 10000  # Ao invés de 5000

# Solução 2: Relaxar critérios temporariamente (CUIDADO!)
MIN_TRADES_REQUIRED = 15  # Só para testes iniciais
```

**Problema**: "SQN sempre baixo"
```python
# Análise: Sistema pode ter alto DD ou muita variação nos trades
# Solução: Ajustar parâmetros para maior consistência
params['sl_atr_multiplier'] = 2.0  # Reduzir SL
params['tp_mult'] = 3.0  # Aumentar TP
```
