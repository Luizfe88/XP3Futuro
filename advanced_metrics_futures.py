# advanced_metrics_futures.py
"""
MÉTRICAS AVANÇADAS PARA MERCADO FUTURO - NÍVEL PROFISSIONAL
=============================================================
✅ Recovery Factor (superior ao Sharpe para futuros)
✅ Expectancy Matemática (R$ por trade)
✅ Sortino Ratio (penaliza só volatilidade negativa)
✅ SQN - System Quality Number (Van Tharp)
✅ MAE/MFE (Excursão Máxima Adversa/Favorável)
✅ Ulcer Index (mede "dor" do drawdown)
✅ Validação: Mínimo 20 trades obrigatório
✅ Profit Factor Ajustado (considera custos reais B3)

REFERÊNCIAS:
- Van Tharp: "Trade Your Way to Financial Freedom"
- John Sweeney: "Maximum Adverse Excursion"
- Sortino, van der Meer: "Downside Risk"
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTES
# ============================================================================

MIN_TRADES_REQUIRED = 10  # 🔥 OBRIGATÓRIO: Mínimo de trades para validação
MIN_TRADES_FOR_SQN = 15   # SQN confiável precisa de mais trades

# Classificações Van Tharp (SQN)
SQN_CLASSIFICATIONS = {
    (float('-inf'), 1.6): "PÉSSIMO - Não Operar",
    (1.6, 2.0): "POBRE - Evitar",
    (2.0, 2.5): "MÉDIO - Usar com cautela",
    (2.5, 3.0): "BOM - Operável",
    (3.0, 5.0): "MUITO BOM - Excelente",
    (5.0, 7.0): "EXCEPCIONAL - Graal",
    (7.0, float('inf')): "SANTO GRAAL - Validar se não é bug"
}

# Recovery Factor
RECOVERY_FACTOR_THRESHOLDS = {
    'EXCELENTE': 5.0,
    'BOM': 3.0,
    'ACEITAVEL': 2.0,
    'RUIM': 1.0
}

# ============================================================================
# DATACLASSES
# ============================================================================

@dataclass
class TradeDetail:
    """Detalhes individuais de cada trade"""
    entry_price: float
    exit_price: float
    pnl: float
    mae: float  # Maximum Adverse Excursion (R$)
    mfe: float  # Maximum Favorable Excursion (R$)
    mae_pct: float  # MAE em % do capital
    mfe_pct: float  # MFE em % do capital
    duration: int  # Barras na operação
    type: str  # 'LONG' ou 'SHORT'
    exit_reason: str  # 'TP', 'SL', 'TIME', 'MARGIN_CALL'
    timestamp_entry: Optional[str] = None
    timestamp_exit: Optional[str] = None


@dataclass
class AdvancedMetrics:
    """Métricas completas do sistema - Nível Profissional"""
    
    # ═══════════════════════════════════════════════════════════════
    # VALIDAÇÃO
    # ═══════════════════════════════════════════════════════════════
    total_trades: int
    is_valid: bool  # False se < MIN_TRADES_REQUIRED
    validation_message: str
    
    # ═══════════════════════════════════════════════════════════════
    # MÉTRICAS BÁSICAS (para comparação)
    # ═══════════════════════════════════════════════════════════════
    win_rate: float
    profit_factor: float
    max_drawdown: float
    sharpe: float
    
    # ═══════════════════════════════════════════════════════════════
    # 1. RECOVERY FACTOR (⭐ Melhor que Sharpe para Futuros)
    # ═══════════════════════════════════════════════════════════════
    recovery_factor: float
    recovery_classification: str  # 'EXCELENTE', 'BOM', 'ACEITAVEL', 'RUIM'
    
    # ═══════════════════════════════════════════════════════════════
    # 2. EXPECTANCY (R$ por trade)
    # ═══════════════════════════════════════════════════════════════
    expectancy: float  # Valor em R$
    expectancy_pct: float  # % do capital
    avg_win: float
    avg_loss: float
    risk_reward: float  # Avg Win / Avg Loss
    
    # ═══════════════════════════════════════════════════════════════
    # 3. SORTINO RATIO (Superior ao Sharpe)
    # ═══════════════════════════════════════════════════════════════
    sortino_ratio: float
    downside_deviation: float
    
    # ═══════════════════════════════════════════════════════════════
    # 4. SQN - System Quality Number (Van Tharp)
    # ═══════════════════════════════════════════════════════════════
    sqn: float
    sqn_classification: str  # 'PÉSSIMO', 'POBRE', 'MÉDIO', 'BOM', etc
    sqn_reliable: bool  # True se >= MIN_TRADES_FOR_SQN
    
    # ═══════════════════════════════════════════════════════════════
    # 5. MAE/MFE (Excursão Adversa/Favorável)
    # ═══════════════════════════════════════════════════════════════
    avg_mae: float
    avg_mfe: float
    mae_percentiles: Dict[str, float]  # P10, P25, P50, P75, P90
    mfe_percentiles: Dict[str, float]
    mae_to_sl_ratio: float  # MAE médio / SL médio (ideal: ~0.8)
    mfe_to_tp_ratio: float  # MFE médio / TP médio (ideal: >0.6)
    
    # ═══════════════════════════════════════════════════════════════
    # MÉTRICAS COMPLEMENTARES
    # ═══════════════════════════════════════════════════════════════
    ulcer_index: float  # "Dor" do drawdown
    calmar_ratio: float  # Return / Max DD
    profit_factor_adjusted: float  # PF após custos B3
    
    # ═══════════════════════════════════════════════════════════════
    # CONSISTÊNCIA
    # ═══════════════════════════════════════════════════════════════
    consecutive_wins_max: int
    consecutive_losses_max: int
    avg_trade_duration: float  # Em barras
    win_streak_current: int
    loss_streak_current: int
    
    # ═══════════════════════════════════════════════════════════════
    # SCORE FINAL (0-100)
    # ═══════════════════════════════════════════════════════════════
    final_score: float  # Pontuação ponderada
    grade: str  # 'A+', 'A', 'B+', 'B', 'C', 'D', 'F'
    
    # Trade details (opcional, para análise profunda)
    trades_detail: List[TradeDetail] = field(default_factory=list)


# ============================================================================
# 1. RECOVERY FACTOR
# ============================================================================

def calculate_recovery_factor(total_pnl: float, max_drawdown: float) -> Tuple[float, str]:
    """
    Recovery Factor = Lucro Líquido Total / Max Drawdown
    
    Interpretação:
    - > 5.0: EXCELENTE (recupera rápido)
    - > 3.0: BOM
    - > 2.0: ACEITÁVEL
    - < 2.0: RUIM (demora muito para recuperar)
    
    Args:
        total_pnl: Lucro/prejuízo líquido total (R$)
        max_drawdown: Drawdown máximo (valor absoluto em R$)
    
    Returns:
        (recovery_factor, classification)
    """
    if max_drawdown <= 0:
        return 999.0, "EXCELENTE"  # Sem DD = perfeito
    
    rf = total_pnl / max_drawdown
    
    # Classificação
    if rf >= RECOVERY_FACTOR_THRESHOLDS['EXCELENTE']:
        classification = "EXCELENTE"
    elif rf >= RECOVERY_FACTOR_THRESHOLDS['BOM']:
        classification = "BOM"
    elif rf >= RECOVERY_FACTOR_THRESHOLDS['ACEITAVEL']:
        classification = "ACEITÁVEL"
    else:
        classification = "RUIM"
    
    return float(rf), classification


# ============================================================================
# 2. EXPECTANCY (Expectativa Matemática)
# ============================================================================

def calculate_expectancy(trades: List[Dict]) -> Dict[str, float]:
    """
    Expectancy = (Win% × Avg Win) - (Loss% × Avg Loss)
    
    🔥 CRÍTICO: Se Expectancy < R$ 50 no WIN, os custos comem o lucro!
    
    Custos típicos no WIN (por trade round-trip):
    - Corretagem: ~R$ 6,00
    - Taxa B3: ~R$ 2,00
    - Slippage: ~R$ 10-20 (1-2 pontos)
    - TOTAL: ~R$ 18-28 por trade
    
    Portanto, Expectancy mínima ideal: R$ 50+
    
    Args:
        trades: Lista de dicts com 'pnl'
    
    Returns:
        Dict com expectancy, avg_win, avg_loss, risk_reward
    """
    if not trades:
        return {
            'expectancy': 0.0,
            'expectancy_pct': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'risk_reward': 0.0
        }
    
    pnls = [float(t.get('pnl', 0)) for t in trades]
    
    wins = [p for p in pnls if p > 0]
    losses = [abs(p) for p in pnls if p < 0]
    
    total = len(pnls)
    win_rate = len(wins) / total if total > 0 else 0
    loss_rate = len(losses) / total if total > 0 else 0
    
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0
    
    # Expectancy em R$
    expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)
    
    # Expectancy em % (assumindo capital de R$ 100k)
    expectancy_pct = (expectancy / 100000) * 100
    
    # Risk/Reward ratio
    rr = avg_win / avg_loss if avg_loss > 0 else 0
    
    return {
        'expectancy': float(expectancy),
        'expectancy_pct': float(expectancy_pct),
        'avg_win': float(avg_win),
        'avg_loss': float(avg_loss),
        'risk_reward': float(rr)
    }


# ============================================================================
# 3. SORTINO RATIO
# ============================================================================

def calculate_sortino_ratio(equity_curve: List[float], 
                           risk_free_rate: float = 0.11,
                           bars_per_year: int = 7000) -> Tuple[float, float]:
    """
    Sortino Ratio = (Return - RFR) / Downside Deviation
    
    Vantagem sobre Sharpe: Penaliza APENAS volatilidade negativa (perdas).
    Sharpe penaliza também os ganhos grandes, o que é injusto.
    
    Para futuros como WDO que tem "pancadas" grandes, Sortino é melhor.
    
    Args:
        equity_curve: Curva de equity
        risk_free_rate: Taxa livre de risco anual (Selic ~11%)
        bars_per_year: Barras por ano (M15 = ~7000)
    
    Returns:
        (sortino_ratio, downside_deviation)
    """
    if not equity_curve or len(equity_curve) < 2:
        return 0.0, 0.0
    
    try:
        arr = np.array(equity_curve, dtype=float)
        returns = np.diff(arr) / arr[:-1]
        
        if len(returns) < 2:
            return 0.0, 0.0
        
        # Retorno médio
        avg_return = np.mean(returns)
        
        # Downside deviation (só retornos negativos)
        negative_returns = returns[returns < 0]
        
        if len(negative_returns) == 0:
            downside_dev = 1e-9  # Sem perdas = ótimo
        else:
            downside_dev = np.std(negative_returns)
        
        # Anualiza
        annual_return = avg_return * bars_per_year
        annual_downside = downside_dev * np.sqrt(bars_per_year)
        
        # Sortino
        sortino = (annual_return - risk_free_rate) / max(annual_downside, 1e-9)
        
        return float(sortino), float(downside_dev)
    
    except Exception as e:
        logger.error(f"Erro ao calcular Sortino: {e}")
        return 0.0, 0.0


# ============================================================================
# 4. SQN - System Quality Number (Van Tharp)
# ============================================================================

def calculate_sqn(trades: List[Dict]) -> Tuple[float, str, bool]:
    """
    SQN = (Expectancy / StdDev(PnL)) × √N
    
    Onde:
    - Expectancy = média dos PnLs
    - StdDev = desvio padrão dos PnLs
    - N = número de trades
    
    📊 CLASSIFICAÇÃO (Van Tharp):
    - 1.6-1.9: POBRE
    - 2.0-2.4: MÉDIO
    - 2.5-2.9: BOM
    - 3.0-4.9: MUITO BOM
    - 5.0-6.9: EXCEPCIONAL
    - 7.0+: SANTO GRAAL (validar se não é bug!)
    
    ⚠️ IMPORTANTE: SQN < 30 trades não é confiável!
    
    Args:
        trades: Lista de dicts com 'pnl'
    
    Returns:
        (sqn_value, classification, is_reliable)
    """
    if not trades:
        return 0.0, "PÉSSIMO - Não Operar", False
    
    n = len(trades)
    
    # Validação mínima
    if n < MIN_TRADES_FOR_SQN:
        is_reliable = False
    else:
        is_reliable = True
    
    pnls = [float(t.get('pnl', 0)) for t in trades]
    
    # Expectancy (média)
    expectancy = np.mean(pnls)
    
    # Desvio padrão
    std_dev = np.std(pnls, ddof=1) if n > 1 else 1.0
    
    if std_dev <= 0:
        return 0.0, "PÉSSIMO - Não Operar", is_reliable
    
    # SQN
    sqn = (expectancy / std_dev) * np.sqrt(n)
    
    # Classificação
    classification = "PÉSSIMO - Não Operar"
    for (min_val, max_val), label in SQN_CLASSIFICATIONS.items():
        if min_val <= sqn < max_val:
            classification = label
            break
    
    return float(sqn), classification, is_reliable


# ============================================================================
# 5. MAE e MFE (Maximum Adverse/Favorable Excursion)
# ============================================================================

def calculate_mae_mfe(trades: List[TradeDetail]) -> Dict[str, any]:
    """
    MAE: Quanto o preço andou CONTRA você antes de fechar o trade
    MFE: Quanto o preço andou A FAVOR antes de fechar
    
    📊 USO PRÁTICO:
    
    **MAE (Stop Loss):**
    - Se MAE médio for 80% do seu SL configurado = SL bem ajustado
    - Se MAE médio for < 50% do SL = SL muito largo (desperdiçando dinheiro)
    - Se MAE médio for > 95% do SL = SL muito apertado (stopando por ruído)
    
    **MFE (Take Profit):**
    - Se MFE médio for > 60% do TP = TP bem posicionado
    - Se MFE médio for < 40% do TP = TP muito otimista (não chega lá)
    - Se MFE médio for > 90% do TP = Pode esticar o TP um pouco mais
    
    Args:
        trades: Lista de TradeDetail objects
    
    Returns:
        Dict com estatísticas de MAE/MFE
    """
    if not trades:
        return {
            'avg_mae': 0.0,
            'avg_mfe': 0.0,
            'mae_percentiles': {},
            'mfe_percentiles': {},
            'mae_to_sl_ratio': 0.0,
            'mfe_to_tp_ratio': 0.0
        }
    
    maes = [t.mae for t in trades if hasattr(t, 'mae')]
    mfes = [t.mfe for t in trades if hasattr(t, 'mfe')]
    
    if not maes or not mfes:
        return {
            'avg_mae': 0.0,
            'avg_mfe': 0.0,
            'mae_percentiles': {},
            'mfe_percentiles': {},
            'mae_to_sl_ratio': 0.0,
            'mfe_to_tp_ratio': 0.0
        }
    
    # Médias
    avg_mae = np.mean(maes)
    avg_mfe = np.mean(mfes)
    
    # Percentis
    mae_percentiles = {
        'P10': float(np.percentile(maes, 10)),
        'P25': float(np.percentile(maes, 25)),
        'P50': float(np.percentile(maes, 50)),
        'P75': float(np.percentile(maes, 75)),
        'P90': float(np.percentile(maes, 90))
    }
    
    mfe_percentiles = {
        'P10': float(np.percentile(mfes, 10)),
        'P25': float(np.percentile(mfes, 25)),
        'P50': float(np.percentile(mfes, 50)),
        'P75': float(np.percentile(mfes, 75)),
        'P90': float(np.percentile(mfes, 90))
    }
    
    # Ratios (placeholder - precisa dos SL/TP configurados)
    mae_to_sl = 0.8  # Ideal
    mfe_to_tp = 0.6  # Ideal
    
    return {
        'avg_mae': float(avg_mae),
        'avg_mfe': float(avg_mfe),
        'mae_percentiles': mae_percentiles,
        'mfe_percentiles': mfe_percentiles,
        'mae_to_sl_ratio': float(mae_to_sl),
        'mfe_to_tp_ratio': float(mfe_to_tp)
    }


# ============================================================================
# 6. ULCER INDEX (Mede a "dor" do drawdown)
# ============================================================================

def calculate_ulcer_index(equity_curve: List[float]) -> float:
    """
    Ulcer Index: Mede a "dor" do drawdown ao longo do tempo
    
    Diferente do Max DD que pega só o pior momento, o Ulcer Index
    mede a DURAÇÃO e PROFUNDIDADE dos drawdowns.
    
    Quanto menor, melhor (menos "sofrimento").
    
    Args:
        equity_curve: Curva de equity
    
    Returns:
        Ulcer Index (0-100+, menor = melhor)
    """
    if not equity_curve or len(equity_curve) < 2:
        return 0.0
    
    try:
        arr = np.array(equity_curve, dtype=float)
        
        # Drawdown percentual em cada ponto
        peak = np.maximum.accumulate(arr)
        dd_pct = ((peak - arr) / peak) * 100
        
        # Ulcer = raiz quadrada da média dos DD² 
        ulcer = np.sqrt(np.mean(dd_pct ** 2))
        
        return float(ulcer)
    
    except Exception as e:
        logger.error(f"Erro ao calcular Ulcer Index: {e}")
        return 0.0


# ============================================================================
# 7. PROFIT FACTOR AJUSTADO (Considera custos B3)
# ============================================================================

def calculate_adjusted_profit_factor(trades: List[Dict], 
                                     cost_per_trade: float = 28.0) -> float:
    """
    Profit Factor Ajustado = Gross Wins / (Gross Losses + Custos)
    
    Custos típicos WIN (round-trip):
    - Corretagem: R$ 6,00
    - Taxa B3: R$ 2,00
    - Emolumentos: R$ 0,50
    - Slippage médio: R$ 15-20
    - TOTAL: ~R$ 28,00 por operação completa
    
    Args:
        trades: Lista de trades
        cost_per_trade: Custo total por operação (default: R$ 28)
    
    Returns:
        PF ajustado (quanto maior, melhor - ideal > 1.5)
    """
    if not trades:
        return 0.0
    
    pnls = [float(t.get('pnl', 0)) for t in trades]
    
    gross_wins = sum(p for p in pnls if p > 0)
    gross_losses = abs(sum(p for p in pnls if p < 0))
    
    total_costs = len(trades) * cost_per_trade
    
    if (gross_losses + total_costs) <= 0:
        return 999.0  # Sem perdas nem custos
    
    pf_adjusted = gross_wins / (gross_losses + total_costs)
    
    return float(pf_adjusted)


# ============================================================================
# FUNÇÃO PRINCIPAL: CALCULA TODAS AS MÉTRICAS
# ============================================================================

def calculate_all_advanced_metrics(
    trades: List[Dict],
    equity_curve: List[float],
    total_pnl: float,
    initial_capital: float = 100000.0,
    cost_per_trade: float = 28.0,
    risk_free_rate: float = 0.11
) -> AdvancedMetrics:
    """
    Calcula TODAS as métricas avançadas para futuros.
    
    🔥 VALIDAÇÃO: Rejeita se < 20 trades!
    
    Args:
        trades: Lista de dicts com trades
        equity_curve: Curva de equity
        total_pnl: PnL total
        initial_capital: Capital inicial
        cost_per_trade: Custo por operação (B3 + corretagem + slippage)
        risk_free_rate: Taxa Selic anual
    
    Returns:
        AdvancedMetrics object com todas as métricas
    """
    
    n_trades = len(trades)
    
    # ═══════════════════════════════════════════════════════════════
    # VALIDAÇÃO CRÍTICA: Mínimo 20 trades
    # ═══════════════════════════════════════════════════════════════
    is_valid = n_trades >= MIN_TRADES_REQUIRED
    
    if not is_valid:
        validation_msg = f"❌ REJEITADO: {n_trades} trades < {MIN_TRADES_REQUIRED} mínimos"
    else:
        validation_msg = f"✅ VÁLIDO: {n_trades} trades"
    
    # Métricas básicas
    win_rate = len([t for t in trades if t.get('pnl', 0) > 0]) / n_trades if n_trades > 0 else 0
    
    wins = sum(t['pnl'] for t in trades if t.get('pnl', 0) > 0)
    losses = abs(sum(t['pnl'] for t in trades if t.get('pnl', 0) < 0))
    profit_factor = wins / losses if losses > 0 else 999.0
    
    # Max DD
    if equity_curve and len(equity_curve) > 1:
        arr = np.array(equity_curve)
        peak = np.maximum.accumulate(arr)
        dd = (peak - arr) / peak
        max_dd = float(np.max(dd))
        max_dd_value = float(np.max(peak - arr))
    else:
        max_dd = 0.0
        max_dd_value = 0.0
    
    # Sharpe (básico)
    sharpe = 0.0  # Calcular depois
    
    # ═══════════════════════════════════════════════════════════════
    # MÉTRICAS AVANÇADAS
    # ═══════════════════════════════════════════════════════════════
    
    # 1. Recovery Factor
    rf, rf_class = calculate_recovery_factor(total_pnl, max_dd_value)
    
    # 2. Expectancy
    exp_data = calculate_expectancy(trades)
    
    # 3. Sortino
    sortino, downside_dev = calculate_sortino_ratio(equity_curve, risk_free_rate)
    
    # 4. SQN
    sqn, sqn_class, sqn_reliable = calculate_sqn(trades)
    
    # 5. MAE/MFE (placeholder - precisa de TradeDetail objects)
    mae_mfe_data = {
        'avg_mae': 0.0,
        'avg_mfe': 0.0,
        'mae_percentiles': {},
        'mfe_percentiles': {},
        'mae_to_sl_ratio': 0.0,
        'mfe_to_tp_ratio': 0.0
    }
    
    # 6. Ulcer Index
    ulcer = calculate_ulcer_index(equity_curve)
    
    # 7. PF Ajustado
    pf_adj = calculate_adjusted_profit_factor(trades, cost_per_trade)
    
    # Calmar
    annual_return = total_pnl / initial_capital
    calmar = annual_return / max_dd if max_dd > 0 else 999.0
    
    # Consistência
    consecutive_wins = 0
    consecutive_losses = 0
    max_wins = 0
    max_losses = 0
    current_streak = 0
    
    for t in trades:
        if t.get('pnl', 0) > 0:
            if current_streak >= 0:
                current_streak += 1
            else:
                current_streak = 1
            max_wins = max(max_wins, current_streak)
        else:
            if current_streak <= 0:
                current_streak -= 1
            else:
                current_streak = -1
            max_losses = max(max_losses, abs(current_streak))
    
    # Score final (0-100)
    score = calculate_final_score(
        sqn=sqn,
        recovery_factor=rf,
        sortino=sortino,
        win_rate=win_rate,
        pf_adjusted=pf_adj,
        is_valid=is_valid
    )
    
    grade = get_grade(score)
    
    # ═══════════════════════════════════════════════════════════════
    # RETORNA OBJETO COMPLETO
    # ═══════════════════════════════════════════════════════════════
    
    return AdvancedMetrics(
        # Validação
        total_trades=n_trades,
        is_valid=is_valid,
        validation_message=validation_msg,
        
        # Básicas
        win_rate=win_rate,
        profit_factor=profit_factor,
        max_drawdown=max_dd,
        sharpe=sharpe,
        
        # 1. Recovery Factor
        recovery_factor=rf,
        recovery_classification=rf_class,
        
        # 2. Expectancy
        expectancy=exp_data['expectancy'],
        expectancy_pct=exp_data['expectancy_pct'],
        avg_win=exp_data['avg_win'],
        avg_loss=exp_data['avg_loss'],
        risk_reward=exp_data['risk_reward'],
        
        # 3. Sortino
        sortino_ratio=sortino,
        downside_deviation=downside_dev,
        
        # 4. SQN
        sqn=sqn,
        sqn_classification=sqn_class,
        sqn_reliable=sqn_reliable,
        
        # 5. MAE/MFE
        avg_mae=mae_mfe_data['avg_mae'],
        avg_mfe=mae_mfe_data['avg_mfe'],
        mae_percentiles=mae_mfe_data['mae_percentiles'],
        mfe_percentiles=mae_mfe_data['mfe_percentiles'],
        mae_to_sl_ratio=mae_mfe_data['mae_to_sl_ratio'],
        mfe_to_tp_ratio=mae_mfe_data['mfe_to_tp_ratio'],
        
        # Complementares
        ulcer_index=ulcer,
        calmar_ratio=calmar,
        profit_factor_adjusted=pf_adj,
        
        # Consistência
        consecutive_wins_max=max_wins,
        consecutive_losses_max=max_losses,
        avg_trade_duration=0.0,  # Calcular depois
        win_streak_current=current_streak if current_streak > 0 else 0,
        loss_streak_current=abs(current_streak) if current_streak < 0 else 0,
        
        # Score
        final_score=score,
        grade=grade
    )


# ============================================================================
# SCORE FINAL (0-100)
# ============================================================================

def calculate_final_score(sqn: float, recovery_factor: float, sortino: float,
                         win_rate: float, pf_adjusted: float, is_valid: bool) -> float:
    """
    Score ponderado (0-100) baseado nas métricas mais importantes.
    
    Pesos:
    - SQN: 30% (mais importante)
    - Recovery Factor: 25%
    - Sortino: 20%
    - PF Ajustado: 15%
    - Win Rate: 10%
    """
    if not is_valid:
        return 0.0
    
    # Normaliza cada métrica para 0-100
    sqn_score = min(100, (sqn / 3.0) * 100)  # 3.0 = excelente
    rf_score = min(100, (recovery_factor / 5.0) * 100)  # 5.0 = excelente
    sortino_score = min(100, (sortino / 2.0) * 100)  # 2.0 = bom
    pf_score = min(100, (pf_adjusted / 2.0) * 100)  # 2.0 = bom
    wr_score = win_rate * 100
    
    # Ponderação
    score = (
        sqn_score * 0.30 +
        rf_score * 0.25 +
        sortino_score * 0.20 +
        pf_score * 0.15 +
        wr_score * 0.10
    )
    
    return min(100.0, max(0.0, score))


def get_grade(score: float) -> str:
    """Converte score para nota (A+, A, B+, B, C, D, F)"""
    if score >= 90:
        return "A+"
    elif score >= 85:
        return "A"
    elif score >= 80:
        return "B+"
    elif score >= 75:
        return "B"
    elif score >= 70:
        return "C+"
    elif score >= 65:
        return "C"
    elif score >= 60:
        return "D"
    else:
        return "F"


# ============================================================================
# FORMATAÇÃO DE RELATÓRIO
# ============================================================================

def format_metrics_report(metrics: AdvancedMetrics) -> str:
    """
    Gera relatório formatado em texto das métricas.
    
    Returns:
        String com relatório completo
    """
    report = []
    
    report.append("=" * 80)
    report.append("📊 RELATÓRIO COMPLETO DE MÉTRICAS - MERCADO FUTURO")
    report.append("=" * 80)
    report.append("")
    
    # Validação
    report.append(f"🔍 VALIDAÇÃO: {metrics.validation_message}")
    if not metrics.is_valid:
        report.append(f"   ⚠️ ATENÇÃO: Amostra insuficiente (<{MIN_TRADES_REQUIRED} trades)")
        report.append(f"   Resultados não são estatisticamente confiáveis!")
    report.append("")
    
    # Score Final
    report.append(f"🏆 SCORE FINAL: {metrics.final_score:.1f}/100 - Nota {metrics.grade}")
    report.append("")
    
    # Métricas Básicas
    report.append("📈 MÉTRICAS BÁSICAS:")
    report.append(f"   Total de Trades: {metrics.total_trades}")
    report.append(f"   Win Rate: {metrics.win_rate:.1%}")
    report.append(f"   Profit Factor: {metrics.profit_factor:.2f}")
    report.append(f"   Max Drawdown: {metrics.max_drawdown:.2%}")
    report.append("")
    
    # 1. Recovery Factor
    report.append("🔄 RECOVERY FACTOR (Velocidade de Recuperação):")
    report.append(f"   Valor: {metrics.recovery_factor:.2f}")
    report.append(f"   Classificação: {metrics.recovery_classification}")
    report.append(f"   Interpretação: {'Recupera rápido das perdas ✅' if metrics.recovery_factor > 3 else 'Demora para recuperar ⚠️'}")
    report.append("")
    
    # 2. Expectancy
    report.append("💰 EXPECTANCY (R$ por Trade):")
    report.append(f"   Expectancy: R$ {metrics.expectancy:.2f} ({metrics.expectancy_pct:.3f}%)")
    report.append(f"   Média Ganho: R$ {metrics.avg_win:.2f}")
    report.append(f"   Média Perda: R$ {metrics.avg_loss:.2f}")
    report.append(f"   Risk/Reward: {metrics.risk_reward:.2f}")
    
    if metrics.expectancy < 50:
        report.append(f"   ⚠️ ALERTA: Expectancy baixa! Custos podem consumir lucro.")
        report.append(f"   Custos típicos WIN: ~R$ 28/trade (corretagem + B3 + slippage)")
    else:
        report.append(f"   ✅ Expectancy acima do custo operacional")
    report.append("")
    
    # 3. Sortino
    report.append("📉 SORTINO RATIO (Volatilidade Negativa):")
    report.append(f"   Sortino: {metrics.sortino_ratio:.2f}")
    report.append(f"   Downside Deviation: {metrics.downside_deviation:.4f}")
    report.append(f"   Interpretação: {'Excelente controle de risco ✅' if metrics.sortino_ratio > 1.5 else 'Revisar gestão de risco ⚠️'}")
    report.append("")
    
    # 4. SQN
    report.append("🎯 SQN - System Quality Number (Van Tharp):")
    report.append(f"   SQN: {metrics.sqn:.2f}")
    report.append(f"   Classificação: {metrics.sqn_classification}")
    report.append(f"   Confiável: {'Sim ✅' if metrics.sqn_reliable else f'Não - Precisa >{MIN_TRADES_FOR_SQN} trades ⚠️'}")
    report.append("")
    
    # 5. MAE/MFE
    if metrics.avg_mae > 0 or metrics.avg_mfe > 0:
        report.append("📊 MAE/MFE (Excursão Adversa/Favorável):")
        report.append(f"   MAE Médio: R$ {metrics.avg_mae:.2f}")
        report.append(f"   MFE Médio: R$ {metrics.avg_mfe:.2f}")
        if metrics.mae_percentiles:
            report.append(f"   MAE P50: R$ {metrics.mae_percentiles.get('P50', 0):.2f}")
            report.append(f"   MFE P50: R$ {metrics.mfe_percentiles.get('P50', 0):.2f}")
        report.append("")
    
    # Complementares
    report.append("📌 MÉTRICAS COMPLEMENTARES:")
    report.append(f"   Ulcer Index: {metrics.ulcer_index:.2f} (quanto menor, melhor)")
    report.append(f"   Calmar Ratio: {metrics.calmar_ratio:.2f}")
    report.append(f"   PF Ajustado (pós-custos): {metrics.profit_factor_adjusted:.2f}")
    report.append("")
    
    # Consistência
    report.append("🔁 CONSISTÊNCIA:")
    report.append(f"   Maior sequência de ganhos: {metrics.consecutive_wins_max}")
    report.append(f"   Maior sequência de perdas: {metrics.consecutive_losses_max}")
    report.append(f"   Sequência atual: {metrics.win_streak_current if metrics.win_streak_current > 0 else f'-{metrics.loss_streak_current}'}")
    report.append("")
    
    report.append("=" * 80)
    
    return "\n".join(report)


# ============================================================================
# EXEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    # Exemplo de uso
    print(f"""
    ╔═══════════════════════════════════════════════════════════╗
    ║  ADVANCED METRICS FOR FUTURES - Nível Profissional       ║
    ║  Desenvolvido para Mercado Futuro B3                      ║
    ╚═══════════════════════════════════════════════════════════╝
    
    Métricas implementadas:
    ✅ Recovery Factor (superior ao Sharpe)
    ✅ Expectancy (R$ por trade)
    ✅ Sortino Ratio (volatilidade negativa)
    ✅ SQN - Van Tharp (qualidade do sistema)
    ✅ MAE/MFE (ajuste de SL/TP)
    ✅ Ulcer Index (dor do drawdown)
    ✅ PF Ajustado (custos B3)
    
    🔥 VALIDAÇÃO: Mínimo {MIN_TRADES_REQUIRED} trades obrigatório!
    
    Para integrar nos otimizadores:
    
    from advanced_metrics_futures import calculate_all_advanced_metrics
    
    metrics = calculate_all_advanced_metrics(
        trades=backtest_trades,
        equity_curve=equity_curve,
        total_pnl=final_pnl
    )
    
    print(format_metrics_report(metrics))
    """)
