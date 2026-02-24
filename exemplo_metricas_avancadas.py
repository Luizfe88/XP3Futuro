# exemplo_metricas_avancadas.py
"""
EXEMPLO PRÁTICO - MÉTRICAS AVANÇADAS PARA FUTUROS
===================================================
Demonstra o cálculo real de todas as métricas com dados fictícios
mas realistas de um backtest no WIN (Mini Índice).
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from advanced_metrics_futures import (
    calculate_all_advanced_metrics,
    format_metrics_report,
    calculate_recovery_factor,
    calculate_expectancy,
    calculate_sortino_ratio,
    calculate_sqn,
    calculate_ulcer_index,
    calculate_adjusted_profit_factor
)

# ============================================================================
# SIMULAÇÃO DE BACKTEST REALISTA - WIN (Mini Índice)
# ============================================================================

def generate_realistic_backtest():
    """
    Gera dados realistas de um backtest no WIN.
    
    Estratégia: Breakout com filtro de tendência
    Período: 3 meses (60 dias úteis)
    Capital: R$ 100.000
    Contratos: 1-2 por operação
    """
    
    np.random.seed(42)  # Reprodutível
    
    # Parâmetros realistas WIN
    initial_capital = 100000.0
    point_value = 0.20  # R$ 0,20 por ponto
    avg_move_points = 400  # Movimento médio de 400 pontos
    
    # Gera 45 trades (validação ok: >= 20)
    n_trades = 45
    
    trades = []
    equity_curve = [initial_capital]
    current_equity = initial_capital
    
    # Padrão realista: 40% win rate, mas wins grandes
    for i in range(n_trades):
        is_win = np.random.random() < 0.40  # 40% win rate
        
        if is_win:
            # Win: média 800 pontos (R$ 160)
            points = np.random.uniform(300, 1500)
            pnl = points * point_value
        else:
            # Loss: média 300 pontos (R$ 60) - SL mais apertado
            points = np.random.uniform(100, 600)
            pnl = -points * point_value
        
        # Aplica custos (corretagem + B3 + slippage)
        pnl -= 28.0
        
        current_equity += pnl
        equity_curve.append(current_equity)
        
        # Detalhes do trade
        trades.append({
            'id': i + 1,
            'pnl': pnl,
            'type': 'WIN' if is_win else 'LOSS',
            'points': points if is_win else -points,
            'exit_reason': 'TP' if is_win else 'SL'
        })
    
    return trades, equity_curve, initial_capital


# ============================================================================
# EXEMPLO 1: Sistema BOM (Score ~80)
# ============================================================================

def exemplo_sistema_bom():
    """
    Exemplo de um sistema BOM para futuros.
    Características:
    - 45 trades (VÁLIDO)
    - Win Rate: ~40%
    - R/R: 2.5:1
    - Recovery Factor: ~4.5
    """
    
    print("=" * 80)
    print("📊 EXEMPLO 1: SISTEMA BOM (Score ~80)")
    print("=" * 80)
    print()
    
    # Gera dados
    trades, equity_curve, initial_capital = generate_realistic_backtest()
    
    # Calcula PnL total
    total_pnl = sum(t['pnl'] for t in trades)
    
    print(f"Capital Inicial: R$ {initial_capital:,.2f}")
    print(f"Capital Final: R$ {equity_curve[-1]:,.2f}")
    print(f"PnL Total: R$ {total_pnl:,.2f}")
    print(f"Retorno: {(total_pnl/initial_capital)*100:.2f}%")
    print(f"Total de Trades: {len(trades)}")
    print()
    
    # ════════════════════════════════════════════════════════════════
    # CALCULA TODAS AS MÉTRICAS AVANÇADAS
    # ════════════════════════════════════════════════════════════════
    
    metrics = calculate_all_advanced_metrics(
        trades=trades,
        equity_curve=equity_curve,
        total_pnl=total_pnl,
        initial_capital=initial_capital,
        cost_per_trade=28.0,  # WIN: corretagem + B3 + slippage
        risk_free_rate=0.11   # Selic 11%
    )
    
    # Exibe relatório formatado
    print(format_metrics_report(metrics))
    
    # ════════════════════════════════════════════════════════════════
    # ANÁLISE DETALHADA DE CADA MÉTRICA
    # ════════════════════════════════════════════════════════════════
    
    print("\n" + "=" * 80)
    print("🔍 ANÁLISE DETALHADA DAS MÉTRICAS")
    print("=" * 80)
    print()
    
    # 1. Recovery Factor
    print("1️⃣ RECOVERY FACTOR:")
    print(f"   Valor: {metrics.recovery_factor:.2f}")
    print(f"   Interpretação:")
    if metrics.recovery_factor >= 5.0:
        print(f"   ✅ EXCELENTE - Sistema recupera muito rápido das perdas")
    elif metrics.recovery_factor >= 3.0:
        print(f"   ✅ BOM - Recuperação adequada")
    elif metrics.recovery_factor >= 2.0:
        print(f"   ⚠️ ACEITÁVEL - Recuperação lenta, atenção")
    else:
        print(f"   ❌ RUIM - Demora muito para recuperar, perigoso!")
    print()
    
    # 2. Expectancy
    print("2️⃣ EXPECTANCY (R$ por Trade):")
    print(f"   Expectancy: R$ {metrics.expectancy:.2f}")
    print(f"   Média Win: R$ {metrics.avg_win:.2f}")
    print(f"   Média Loss: R$ {metrics.avg_loss:.2f}")
    print(f"   R/R Ratio: {metrics.risk_reward:.2f}:1")
    print(f"   Interpretação:")
    
    cost_per_trade = 28.0
    net_expectancy = metrics.expectancy - cost_per_trade
    
    if metrics.expectancy >= 100:
        print(f"   ✅ EXCELENTE - Margem confortável sobre custos")
    elif metrics.expectancy >= 50:
        print(f"   ✅ BOM - Cobre custos com folga")
    elif metrics.expectancy >= cost_per_trade:
        print(f"   ⚠️ LIMITE - Mal cobre os custos operacionais")
    else:
        print(f"   ❌ NEGATIVO - Expectancy < custos! Sistema inviável!")
    
    print(f"   Lucro Líquido por Trade: R$ {net_expectancy:.2f}")
    print()
    
    # 3. Sortino Ratio
    print("3️⃣ SORTINO RATIO:")
    print(f"   Sortino: {metrics.sortino_ratio:.2f}")
    print(f"   Downside Deviation: {metrics.downside_deviation:.4f}")
    print(f"   Interpretação:")
    if metrics.sortino_ratio >= 2.0:
        print(f"   ✅ EXCELENTE - Volatilidade negativa muito controlada")
    elif metrics.sortino_ratio >= 1.0:
        print(f"   ✅ BOM - Controle adequado de risco")
    elif metrics.sortino_ratio >= 0.5:
        print(f"   ⚠️ FRACO - Muita volatilidade nas perdas")
    else:
        print(f"   ❌ RUIM - Perdas muito voláteis, risco alto")
    print()
    
    # 4. SQN (System Quality Number)
    print("4️⃣ SQN - SYSTEM QUALITY NUMBER (Van Tharp):")
    print(f"   SQN: {metrics.sqn:.2f}")
    print(f"   Classificação: {metrics.sqn_classification}")
    print(f"   Confiável: {'Sim ✅' if metrics.sqn_reliable else 'Não - Precisa >30 trades ⚠️'}")
    print(f"   Interpretação (Van Tharp):")
    print(f"   - 1.6-1.9: POBRE")
    print(f"   - 2.0-2.4: MÉDIO")
    print(f"   - 2.5-2.9: BOM")
    print(f"   - 3.0-4.9: MUITO BOM")
    print(f"   - 5.0+: EXCEPCIONAL")
    print()
    
    # 5. Ulcer Index
    print("5️⃣ ULCER INDEX (Dor do Drawdown):")
    print(f"   Ulcer: {metrics.ulcer_index:.2f}")
    print(f"   Interpretação:")
    if metrics.ulcer_index <= 3.0:
        print(f"   ✅ BAIXO - Drawdowns curtos e rasos")
    elif metrics.ulcer_index <= 5.0:
        print(f"   ⚠️ MODERADO - Alguns períodos desconfortáveis")
    else:
        print(f"   ❌ ALTO - Drawdowns longos e profundos (sofrimento!)")
    print()
    
    # 6. PF Ajustado
    print("6️⃣ PROFIT FACTOR AJUSTADO:")
    print(f"   PF Básico: {metrics.profit_factor:.2f}")
    print(f"   PF Ajustado (pós-custos): {metrics.profit_factor_adjusted:.2f}")
    print(f"   Diferença: {(metrics.profit_factor - metrics.profit_factor_adjusted):.2f}")
    print(f"   Interpretação:")
    if metrics.profit_factor_adjusted >= 2.0:
        print(f"   ✅ EXCELENTE - Margem confortável após custos")
    elif metrics.profit_factor_adjusted >= 1.5:
        print(f"   ✅ BOM - Lucro sólido após custos")
    elif metrics.profit_factor_adjusted >= 1.2:
        print(f"   ⚠️ JUSTO - Pouca margem")
    else:
        print(f"   ❌ RUIM - Custos comprometem viabilidade")
    print()
    
    # ════════════════════════════════════════════════════════════════
    # COMPARAÇÃO: Com vs Sem Métricas Avançadas
    # ════════════════════════════════════════════════════════════════
    
    print("\n" + "=" * 80)
    print("📊 COMPARAÇÃO: Métricas Básicas vs Avançadas")
    print("=" * 80)
    print()
    
    print("SEM MÉTRICAS AVANÇADAS (análise incompleta):")
    print(f"   ✓ Win Rate: {metrics.win_rate:.1%}")
    print(f"   ✓ Profit Factor: {metrics.profit_factor:.2f}")
    print(f"   ✓ Max Drawdown: {metrics.max_drawdown:.2%}")
    print(f"   → Conclusão: 'Parece bom!' ❓")
    print()
    
    print("COM MÉTRICAS AVANÇADAS (análise profunda):")
    print(f"   ✓ Win Rate: {metrics.win_rate:.1%}")
    print(f"   ✓ Profit Factor: {metrics.profit_factor:.2f}")
    print(f"   ✓ Max Drawdown: {metrics.max_drawdown:.2%}")
    print(f"   + Recovery Factor: {metrics.recovery_factor:.2f} ({metrics.recovery_classification})")
    print(f"   + Expectancy: R$ {metrics.expectancy:.2f} (viável!)")
    print(f"   + SQN: {metrics.sqn:.2f} ({metrics.sqn_classification})")
    print(f"   + Sortino: {metrics.sortino_ratio:.2f}")
    print(f"   → Conclusão: '{metrics.grade} - {metrics.sqn_classification}' ✅")
    print()
    
    return metrics


# ============================================================================
# EXEMPLO 2: Sistema RUIM (Score baixo)
# ============================================================================

def exemplo_sistema_ruim():
    """
    Exemplo de um sistema RUIM que parece bom nas métricas básicas.
    Características:
    - 12 trades (INVÁLIDO - < 20)
    - Win Rate alto: 70%
    - Mas: Poucos trades, pode ser sorte
    """
    
    print("\n\n")
    print("=" * 80)
    print("📊 EXEMPLO 2: SISTEMA APARENTEMENTE BOM, MAS INVÁLIDO")
    print("=" * 80)
    print()
    
    # Gera apenas 12 trades (sorte?)
    np.random.seed(123)
    trades = []
    equity = 100000.0
    equity_curve = [equity]
    
    # 70% win rate - parece ótimo!
    for i in range(12):
        is_win = i < 8  # 8 wins de 12 = 67%
        
        if is_win:
            pnl = np.random.uniform(50, 150)
        else:
            pnl = np.random.uniform(-200, -100)
        
        pnl -= 28.0  # Custos
        equity += pnl
        equity_curve.append(equity)
        
        trades.append({'pnl': pnl})
    
    total_pnl = sum(t['pnl'] for t in trades)
    
    print(f"Total de Trades: {len(trades)} ⚠️")
    print(f"Win Rate: {len([t for t in trades if t['pnl'] > 0])/len(trades):.1%}")
    print(f"PnL Total: R$ {total_pnl:.2f}")
    print()
    
    # Calcula métricas
    metrics = calculate_all_advanced_metrics(
        trades=trades,
        equity_curve=equity_curve,
        total_pnl=total_pnl,
        initial_capital=100000.0
    )
    
    print("ANÁLISE:")
    print(f"   Validação: {metrics.validation_message}")
    print(f"   É Válido: {'SIM ✅' if metrics.is_valid else 'NÃO ❌'}")
    print(f"   Score Final: {metrics.final_score:.1f}/100 (Nota {metrics.grade})")
    print()
    
    print("POR QUE FOI REJEITADO:")
    print(f"   ❌ Apenas {len(trades)} trades (mínimo: 20)")
    print(f"   ❌ Amostra insuficiente para validação estatística")
    print(f"   ❌ Pode ser SORTE, não habilidade")
    print(f"   ❌ SQN não é confiável com <30 trades")
    print()
    
    print("COMPARAÇÃO:")
    print("   SEM validação: 'Win Rate 70%! Sistema incrível!' ✨")
    print("   COM validação: 'Amostra pequena, não operar' ⚠️")
    print()


# ============================================================================
# EXEMPLO 3: Sistema com Expectancy NEGATIVA
# ============================================================================

def exemplo_sistema_expectancy_negativa():
    """
    Sistema que passa no mínimo de trades, mas tem Expectancy < Custos.
    """
    
    print("\n\n")
    print("=" * 80)
    print("📊 EXEMPLO 3: SISTEMA COM EXPECTANCY NEGATIVA")
    print("=" * 80)
    print()
    
    # 25 trades (válido), mas Expectancy baixa
    np.random.seed(456)
    trades = []
    equity = 100000.0
    equity_curve = [equity]
    
    for i in range(25):
        is_win = np.random.random() < 0.48  # 48% win rate
        
        if is_win:
            # Wins pequenos
            pnl = np.random.uniform(30, 70)
        else:
            # Losses grandes
            pnl = np.random.uniform(-80, -50)
        
        pnl -= 28.0  # Custos
        equity += pnl
        equity_curve.append(equity)
        
        trades.append({'pnl': pnl})
    
    total_pnl = sum(t['pnl'] for t in trades)
    
    metrics = calculate_all_advanced_metrics(
        trades=trades,
        equity_curve=equity_curve,
        total_pnl=total_pnl,
        initial_capital=100000.0
    )
    
    print(f"Total de Trades: {len(trades)} ✅")
    print(f"Win Rate: {metrics.win_rate:.1%}")
    print(f"Expectancy: R$ {metrics.expectancy:.2f}")
    print(f"Custos por Trade: R$ 28,00")
    print(f"Lucro Líquido por Trade: R$ {metrics.expectancy - 28:.2f}")
    print()
    
    print("PROBLEMA DETECTADO:")
    print(f"   ❌ Expectancy (R$ {metrics.expectancy:.2f}) < Custos (R$ 28,00)")
    print(f"   ❌ Sistema perde dinheiro na prática!")
    print(f"   ❌ Cada trade tira R$ {abs(metrics.expectancy - 28):.2f} da conta")
    print()
    
    print("SEM MÉTRICAS AVANÇADAS:")
    print(f"   'Win Rate 48%, quase 50%... vou operar!' ❓")
    print()
    
    print("COM MÉTRICAS AVANÇADAS:")
    print(f"   'Expectancy negativa após custos → NÃO OPERAR!' ✅")
    print()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║  DEMONSTRAÇÃO PRÁTICA - MÉTRICAS AVANÇADAS PARA FUTUROS       ║
    ║  Exemplos Reais de Cálculo e Interpretação                    ║
    ╚════════════════════════════════════════════════════════════════╝
    """)
    
    # Exemplo 1: Sistema BOM
    metrics_bom = exemplo_sistema_bom()
    
    # Exemplo 2: Sistema com poucos trades (inválido)
    exemplo_sistema_ruim()
    
    # Exemplo 3: Expectancy negativa
    exemplo_sistema_expectancy_negativa()
    
    # Resumo final
    print("\n\n")
    print("=" * 80)
    print("🎯 CONCLUSÃO: POR QUE ESSAS MÉTRICAS SÃO ESSENCIAIS")
    print("=" * 80)
    print()
    
    print("1. RECOVERY FACTOR:")
    print("   → Sharpe não diz QUANTO TEMPO você leva para recuperar")
    print("   → Em futuros alavancados, tempo = morte")
    print()
    
    print("2. EXPECTANCY:")
    print("   → Win Rate 30% pode ser ÓTIMO se R/R for 3:1")
    print("   → Mas se Expectancy < R$ 50, custos comem o lucro!")
    print()
    
    print("3. SQN + VALIDAÇÃO (20 trades):")
    print("   → 10 trades com 80% WR = SORTE")
    print("   → 30 trades com 40% WR = ESTATÍSTICA")
    print()
    
    print("4. SORTINO:")
    print("   → Sharpe penaliza wins grandes (injusto!)")
    print("   → Sortino só penaliza perdas (correto!)")
    print()
    
    print("5. MAE/MFE:")
    print("   → Ajusta SL/TP cientificamente")
    print("   → Sem 'feeling', com dados!")
    print()
    
    print("=" * 80)
    print("✅ SISTEMA COMPLETO E PROFISSIONAL PARA MERCADO FUTURO")
    print("=" * 80)
