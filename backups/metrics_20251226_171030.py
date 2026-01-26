"""
=============================================================================
MÓDULO DE MÉTRICAS UNIFICADO - XP3 PRO BOT
=============================================================================
Sistema completo de cálculo de performance com métricas padrão da indústria.

Características:
- Cálculos matematicamente corretos (Sharpe, Sortino, Calmar corrigidos)
- Métricas de risco avançadas (VaR, CVaR, Ulcer Index)
- Análise de drawdown detalhada
- Métricas de trading (streaks, exposure, R-multiples)
- Validações robustas e tratamento de erros
- Compatível com backtest e live trading

Autor: XP3 Pro Bot Team
Data: Dezembro 2024
=============================================================================
"""

import numpy as np
import pandas as pd
import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger("metrics")

# =============================================================================
# CONFIGURAÇÕES
# =============================================================================

@dataclass
class MetricsConfig:
    """Configurações para cálculo de métricas"""
    
    bars_per_day: int = 28  # M15: 28 barras/dia (10:00-17:00)
    trading_days_per_year: int = 252
    risk_free_rate: float = 0.1075  # CDI ~10.75% a.a. (2024)
    
    # Janelas de tempo
    calmar_window_months: int = 36  # Calmar usa 36 meses
    
    # Percentis para VaR
    var_confidence_levels: List[float] = None
    
    # Thresholds
    min_trades_for_metrics: int = 10
    min_bars_for_annual: int = 252  # 1 ano de dados mínimo
    
    def __post_init__(self):
        if self.var_confidence_levels is None:
            self.var_confidence_levels = [0.90, 0.95, 0.99]
    
    @property
    def bars_per_year(self) -> int:
        return self.bars_per_day * self.trading_days_per_year


# Instância global padrão
DEFAULT_CONFIG = MetricsConfig()


# =============================================================================
# CLASSE PRINCIPAL DE MÉTRICAS
# =============================================================================

@dataclass
class PerformanceMetrics:
    """
    Classe unificada para armazenar todas as métricas de performance.
    """
    
    # ===== RETORNOS ===== (todos sem default)
    total_return: float
    annualized_return: float
    monthly_return: float
    cagr: float
    
    # ===== RISCO ===== (todos sem default)
    volatility_annual: float
    max_drawdown: float
    avg_drawdown: float
    max_dd_duration_days: float
    current_drawdown: float
    
    # ===== RATIOS ===== (todos sem default)
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    omega_ratio: float
    ulcer_index: float
    
    # ===== TRADING METRICS ===== (todos sem default)
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    expectancy: float
    avg_win: float
    avg_loss: float
    avg_r_multiple: float
    largest_win: float
    largest_loss: float
    
    # ===== STREAKS ===== (todos sem default)
    max_consecutive_wins: int
    max_consecutive_losses: int
    current_streak: int
    current_streak_type: str  # "WIN" ou "LOSS"
    
    # ===== EXPOSURE ===== (todos sem default)
    exposure_pct: float
    avg_trade_duration_hours: float
    trades_per_day: float
    
    # ===== RISK METRICS ===== (sem default primeiro)
    var_95: float
    cvar_95: float
    
    # ===== RECOVERY ===== (sem default primeiro)
    recovery_factor: float
    
    # ===== EQUITY ===== (sem default primeiro)
    initial_equity: float = 100000.0
    final_equity: float = 100000.0
    peak_equity: float = 100000.0
    
    # ===== CAMPOS COM DEFAULT (todos no final) =====
    var_99: Optional[float] = None
    cvar_99: Optional[float] = None
    recovery_time_days: Optional[float] = None
    
    avg_mae: Optional[float] = None
    avg_mfe: Optional[float] = None
    mae_to_profit_ratio: Optional[float] = None
    
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    days_in_market: Optional[int] = None
    
    # ===== EQUITY =====
    initial_equity: float = 100000.0
    final_equity: float = 100000.0
    peak_equity: float = 100000.0
    
    def to_dict(self) -> Dict:
        """Converte para dicionário"""
        return asdict(self)
    
    def to_json(self) -> str:
        """Serializa para JSON"""
        import json
        return json.dumps(self.to_dict(), indent=2, default=str)
    
    def __str__(self) -> str:
        """Formatação para exibição no console"""
        
        # Emoji para indicadores
        return_emoji = "🟢" if self.total_return > 0 else "🔴"
        sharpe_emoji = "🟢" if self.sharpe_ratio > 1.5 else "🟡" if self.sharpe_ratio > 1.0 else "🔴"
        wr_emoji = "🟢" if self.win_rate > 0.55 else "🟡" if self.win_rate > 0.45 else "🔴"
        
        return f"""
╔══════════════════════════════════════════════════════════╗
║          PERFORMANCE METRICS REPORT - XP3 PRO            ║
╠══════════════════════════════════════════════════════════╣
║ PERÍODO                                                  ║
║  • Início: {self.start_date or 'N/A':>44} ║
║  • Fim: {self.end_date or 'N/A':>47} ║
║  • Dias: {self.days_in_market or 0:>46} ║
╠══════════════════════════════════════════════════════════╣
║ RETORNOS {return_emoji}                                           ║
║  • Total: {self.total_return:>43.2%} ║
║  • Anualizado (CAGR): {self.cagr:>33.2%} ║
║  • Mensal: {self.monthly_return:>42.2%} ║
║  • Equity Final: R$ {self.final_equity:>34,.2f} ║
╠══════════════════════════════════════════════════════════╣
║ RISCO                                                    ║
║  • Max Drawdown: {self.max_drawdown:>36.2%} ║
║  • Drawdown Atual: {self.current_drawdown:>34.2%} ║
║  • DD Médio: {self.avg_drawdown:>40.2%} ║
║  • Maior DD (dias): {self.max_dd_duration_days:>33.1f} ║
║  • Volatilidade Anual: {self.volatility_annual:>28.2%} ║
║  • VaR 95%: {self.var_95:>41.2%} ║
║  • CVaR 95%: {self.cvar_95:>40.2%} ║
╠══════════════════════════════════════════════════════════╣
║ RATIOS {sharpe_emoji}                                             ║
║  • Sharpe: {self.sharpe_ratio:>42.2f} ║
║  • Sortino: {self.sortino_ratio:>41.2f} ║
║  • Calmar: {self.calmar_ratio:>42.2f} ║
║  • Omega: {self.omega_ratio:>43.2f} ║
║  • Ulcer Index: {self.ulcer_index:>37.2f} ║
╠══════════════════════════════════════════════════════════╣
║ TRADING {wr_emoji}                                                ║
║  • Total Trades: {self.total_trades:>38} ║
║  • Vencedores: {self.winning_trades:>40} ║
║  • Perdedores: {self.losing_trades:>40} ║
║  • Win Rate: {self.win_rate:>40.1%} ║
║  • Profit Factor: {self.profit_factor:>35.2f} ║
║  • Expectancy: R$ {self.expectancy:>34,.2f} ║
║  • Avg Win: R$ {self.avg_win:>38,.2f} ║
║  • Avg Loss: R$ {abs(self.avg_loss):>37,.2f} ║
║  • R-Multiple Médio: {self.avg_r_multiple:>30.2f} ║
╠══════════════════════════════════════════════════════════╣
║ STREAKS                                                  ║
║  • Máx Vitórias Consecutivas: {self.max_consecutive_wins:>23} ║
║  • Máx Perdas Consecutivas: {self.max_consecutive_losses:>25} ║
║  • Streak Atual: {self.current_streak:>37} {self.current_streak_type:<4} ║
╠══════════════════════════════════════════════════════════╣
║ EXPOSURE & EFICIÊNCIA                                    ║
║  • Tempo em Posição: {self.exposure_pct:>32.1f}% ║
║  • Duração Média Trade: {self.avg_trade_duration_hours:>25.1f}h ║
║  • Trades/Dia: {self.trades_per_day:>38.2f} ║
╠══════════════════════════════════════════════════════════╣
║ RECOVERY                                                 ║
║  • Recovery Factor: {self.recovery_factor:>33.2f} ║
║  • Recovery Time: {self.recovery_time_days or 0:>33.1f} dias ║
╚══════════════════════════════════════════════════════════╝
        """.strip()
    
    def get_grade(self) -> str:
        """
        Retorna nota de A+ a F baseada em múltiplos critérios
        
        Critérios:
        - Sharpe > 2.0: Excelente
        - Win Rate > 55%: Bom
        - Profit Factor > 2.0: Bom
        - Max DD < 15%: Aceitável
        """
        score = 0
        
        # Sharpe (0-30 pontos)
        if self.sharpe_ratio >= 2.5:
            score += 30
        elif self.sharpe_ratio >= 2.0:
            score += 25
        elif self.sharpe_ratio >= 1.5:
            score += 20
        elif self.sharpe_ratio >= 1.0:
            score += 15
        
        # Win Rate (0-25 pontos)
        if self.win_rate >= 0.60:
            score += 25
        elif self.win_rate >= 0.55:
            score += 20
        elif self.win_rate >= 0.50:
            score += 15
        elif self.win_rate >= 0.45:
            score += 10
        
        # Profit Factor (0-25 pontos)
        if self.profit_factor >= 2.5:
            score += 25
        elif self.profit_factor >= 2.0:
            score += 20
        elif self.profit_factor >= 1.5:
            score += 15
        elif self.profit_factor >= 1.2:
            score += 10
        
        # Max Drawdown (0-20 pontos)
        if self.max_drawdown <= 0.10:
            score += 20
        elif self.max_drawdown <= 0.15:
            score += 15
        elif self.max_drawdown <= 0.20:
            score += 10
        elif self.max_drawdown <= 0.25:
            score += 5
        
        # Converte score para nota
        if score >= 85:
            return "A+"
        elif score >= 80:
            return "A"
        elif score >= 75:
            return "A-"
        elif score >= 70:
            return "B+"
        elif score >= 65:
            return "B"
        elif score >= 60:
            return "B-"
        elif score >= 55:
            return "C+"
        elif score >= 50:
            return "C"
        elif score >= 45:
            return "C-"
        elif score >= 40:
            return "D"
        else:
            return "F"


# =============================================================================
# FUNÇÕES DE CÁLCULO - RETORNOS
# =============================================================================

def calculate_returns(equity_curve: np.ndarray, config: MetricsConfig = DEFAULT_CONFIG) -> Dict[str, float]:
    """
    Calcula métricas de retorno
    
    Args:
        equity_curve: Array com valores de equity
        config: Configuração de métricas
    
    Returns:
        Dict com total_return, annualized_return, monthly_return, cagr
    """
    
    if len(equity_curve) < 2:
        return {
            'total_return': 0.0,
            'annualized_return': 0.0,
            'monthly_return': 0.0,
            'cagr': 0.0
        }
    
    initial = equity_curve[0]
    final = equity_curve[-1]
    
    total_return = (final / initial) - 1
    
    # Calcula período em anos
    n_bars = len(equity_curve)
    years = n_bars / config.bars_per_year
    
    # CAGR (Compound Annual Growth Rate)
    if years > 0 and final > 0 and initial > 0:
        cagr = (final / initial) ** (1 / years) - 1
        annualized_return = cagr
    else:
        cagr = 0.0
        annualized_return = 0.0
    
    # Retorno mensal
    months = years * 12
    if months > 0:
        monthly_return = (1 + total_return) ** (1 / months) - 1
    else:
        monthly_return = 0.0
    
    return {
        'total_return': float(total_return),
        'annualized_return': float(annualized_return),
        'monthly_return': float(monthly_return),
        'cagr': float(cagr)
    }


# =============================================================================
# FUNÇÕES DE CÁLCULO - DRAWDOWN
# =============================================================================

def calculate_drawdown_series(equity_curve: np.ndarray) -> np.ndarray:
    """
    Calcula série temporal de drawdown
    
    Args:
        equity_curve: Array com valores de equity
    
    Returns:
        Array com drawdowns percentuais (valores negativos)
    """
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / peak
    return drawdown


def calculate_drawdown_metrics(equity_curve: np.ndarray, config: MetricsConfig = DEFAULT_CONFIG) -> Dict[str, float]:
    """
    Calcula métricas detalhadas de drawdown
    
    Args:
        equity_curve: Array com valores de equity
        config: Configuração de métricas
    
    Returns:
        Dict com max_drawdown, avg_drawdown, max_dd_duration, etc
    """
    
    if len(equity_curve) < 2:
        return {
            'max_drawdown': 0.0,
            'avg_drawdown': 0.0,
            'max_dd_duration_days': 0.0,
            'current_drawdown': 0.0,
            'recovery_time_days': None
        }
    
    drawdowns = calculate_drawdown_series(equity_curve)
    
    # Max drawdown
    max_dd = float(abs(np.min(drawdowns)))
    
    # Current drawdown
    current_dd = float(abs(drawdowns[-1]))
    
    # Average drawdown (apenas períodos em DD)
    in_dd = drawdowns < 0
    if np.any(in_dd):
        avg_dd = float(abs(np.mean(drawdowns[in_dd])))
    else:
        avg_dd = 0.0
    
    # Duração de drawdowns
    dd_durations = []
    recovery_times = []
    
    # Identifica períodos de drawdown
    dd_changes = np.diff(in_dd.astype(int))
    dd_starts = np.where(dd_changes == 1)[0] + 1
    dd_ends = np.where(dd_changes == -1)[0] + 1
    
    # Se começar em DD
    if in_dd[0]:
        dd_starts = np.concatenate([[0], dd_starts])
    
    # Se terminar em DD
    if in_dd[-1]:
        dd_ends = np.concatenate([dd_ends, [len(equity_curve)]])
    
    # Calcula durações
    for start, end in zip(dd_starts, dd_ends):
        duration = end - start
        dd_durations.append(duration)
        
        # Calcula recovery time se houver recuperação
        if end < len(equity_curve):
            peak_before = equity_curve[start - 1] if start > 0 else equity_curve[start]
            recovery_idx = np.where(equity_curve[end:] >= peak_before)[0]
            
            if len(recovery_idx) > 0:
                recovery_time = recovery_idx[0]
                recovery_times.append(recovery_time)
    
    # Converte para dias
    max_dd_duration_bars = max(dd_durations) if dd_durations else 0
    max_dd_duration_days = max_dd_duration_bars / config.bars_per_day
    
    # Recovery time médio
    if recovery_times:
        avg_recovery_bars = np.mean(recovery_times)
        recovery_time_days = avg_recovery_bars / config.bars_per_day
    else:
        recovery_time_days = None
    
    return {
        'max_drawdown': max_dd,
        'avg_drawdown': avg_dd,
        'max_dd_duration_days': float(max_dd_duration_days),
        'current_drawdown': current_dd,
        'recovery_time_days': float(recovery_time_days) if recovery_time_days is not None else None
    }


# =============================================================================
# FUNÇÕES DE CÁLCULO - RISK-ADJUSTED RATIOS
# =============================================================================

def calculate_sharpe_ratio(
    equity_curve: np.ndarray,
    risk_free_rate: float = DEFAULT_CONFIG.risk_free_rate,
    config: MetricsConfig = DEFAULT_CONFIG
) -> float:
    """
    ✅ SHARPE RATIO CORRIGIDO
    
    Fórmula correta:
    Sharpe = (Retorno Anualizado - Taxa Livre de Risco) / Volatilidade Anualizada
    
    Args:
        equity_curve: Array com valores de equity
        risk_free_rate: Taxa livre de risco anual
        config: Configuração de métricas
    
    Returns:
        Sharpe Ratio
    """
    
    if len(equity_curve) < config.min_bars_for_annual:
        logger.warning("Sharpe: Dados insuficientes para anualização confiável")
    
    # Calcula retornos
    returns = np.diff(equity_curve) / equity_curve[:-1]
    
    if len(returns) < 2:
        return 0.0
    
    # Retorno médio por período
    mean_return = np.mean(returns)
    
    # Anualiza retorno
    annualized_return = mean_return * config.bars_per_year
    
    # Volatilidade por período
    std_return = np.std(returns, ddof=1)  # ddof=1 para sample std
    
    # ✅ CORREÇÃO: Anualiza a volatilidade
    annualized_volatility = std_return * np.sqrt(config.bars_per_year)
    
    # Sharpe Ratio
    if annualized_volatility == 0:
        return 0.0
    
    sharpe = (annualized_return - risk_free_rate) / annualized_volatility
    
    return float(sharpe)


def calculate_sortino_ratio(
    equity_curve: np.ndarray,
    risk_free_rate: float = DEFAULT_CONFIG.risk_free_rate,
    config: MetricsConfig = DEFAULT_CONFIG
) -> float:
    """
    ✅ SORTINO RATIO CORRIGIDO
    
    Similar ao Sharpe, mas usa apenas downside deviation
    
    Args:
        equity_curve: Array com valores de equity
        risk_free_rate: Taxa livre de risco anual
        config: Configuração de métricas
    
    Returns:
        Sortino Ratio
    """
    
    if len(equity_curve) < config.min_bars_for_annual:
        logger.warning("Sortino: Dados insuficientes para anualização confiável")
    
    returns = np.diff(equity_curve) / equity_curve[:-1]
    
    if len(returns) < 2:
        return 0.0
    
    mean_return = np.mean(returns)
    annualized_return = mean_return * config.bars_per_year
    
    # ✅ CORREÇÃO: Downside deviation (apenas retornos negativos)
    downside_returns = returns[returns < 0]
    
    if len(downside_returns) == 0:
        # Sem perdas = Sortino infinito
        return float('inf') if annualized_return > risk_free_rate else 0.0
    
    downside_std = np.std(downside_returns, ddof=1)
    
    # ✅ CORREÇÃO: Anualiza downside deviation
    annualized_downside_std = downside_std * np.sqrt(config.bars_per_year)
    
    if annualized_downside_std == 0:
        return 0.0
    
    sortino = (annualized_return - risk_free_rate) / annualized_downside_std
    
    return float(sortino)


def calculate_calmar_ratio(
    equity_curve: np.ndarray,
    config: MetricsConfig = DEFAULT_CONFIG
) -> float:
    """
    ✅ CALMAR RATIO CORRIGIDO
    
    Usa janela móvel de 36 meses (padrão da indústria)
    
    Calmar = Retorno Anualizado (36M) / Max Drawdown (36M)
    
    Args:
        equity_curve: Array com valores de equity
        config: Configuração de métricas
    
    Returns:
        Calmar Ratio
    """
    
    if len(equity_curve) < 2:
        return 0.0
    
    # Calcula janela de 36 meses em barras
    bars_in_window = config.calmar_window_months * 21 * config.bars_per_day
    
    # Se não tiver 36 meses, usa todo o período
    if len(equity_curve) < bars_in_window:
        window_equity = equity_curve
        years = len(equity_curve) / config.bars_per_year
    else:
        window_equity = equity_curve[-bars_in_window:]
        years = config.calmar_window_months / 12
    
    # Retorno do período
    total_return = (window_equity[-1] / window_equity[0]) - 1
    
    # Anualiza
    if years > 0:
        annualized_return = (1 + total_return) ** (1 / years) - 1
    else:
        annualized_return = 0.0
    
    # Max drawdown do período
    peak = np.maximum.accumulate(window_equity)
    drawdowns = (window_equity - peak) / peak
    max_dd = abs(np.min(drawdowns))
    
    # Evita divisão por zero
    if max_dd < 0.001:
        max_dd = 0.001
    
    calmar = annualized_return / max_dd
    
    return float(calmar)


def calculate_omega_ratio(
    equity_curve: np.ndarray,
    threshold: float = 0.0,
    config: MetricsConfig = DEFAULT_CONFIG
) -> float:
    """
    Omega Ratio: Probabilidade de ganhos acima de threshold / perdas abaixo
    
    Args:
        equity_curve: Array com valores de equity
        threshold: Retorno mínimo aceitável (0 = break-even)
        config: Configuração
    
    Returns:
        Omega Ratio
    """
    
    returns = np.diff(equity_curve) / equity_curve[:-1]
    
    if len(returns) < 2:
        return 0.0
    
    # Retornos acima e abaixo do threshold
    gains = returns[returns > threshold] - threshold
    losses = threshold - returns[returns < threshold]
    
    sum_gains = np.sum(gains) if len(gains) > 0 else 0.0
    sum_losses = np.sum(losses) if len(losses) > 0 else 1e-10  # Evita divisão por zero
    
    omega = sum_gains / sum_losses
    
    return float(omega)


def calculate_ulcer_index(equity_curve: np.ndarray, config: MetricsConfig = DEFAULT_CONFIG) -> float:
    """
    Ulcer Index: Medida de stress por drawdown
    
    UI = sqrt(mean(drawdown²))
    
    Args:
        equity_curve: Array com valores de equity
        config: Configuração
    
    Returns:
        Ulcer Index
    """
    
    if len(equity_curve) < 2:
        return 0.0
    
    drawdowns = calculate_drawdown_series(equity_curve)
    
    # Squared drawdowns
    squared_dd = drawdowns ** 2
    
    # Ulcer Index
    ulcer = np.sqrt(np.mean(squared_dd))
    
    return float(ulcer) * 100  # Retorna em %


# =============================================================================
# FUNÇÕES DE CÁLCULO - VAR/CVAR
# =============================================================================

def calculate_var_cvar(
    equity_curve: np.ndarray,
    confidence_levels: List[float] = None,
    config: MetricsConfig = DEFAULT_CONFIG
) -> Dict[str, float]:
    """
    Value at Risk (VaR) e Conditional VaR (CVaR/Expected Shortfall)
    
    Args:
        equity_curve: Array com valores de equity
        confidence_levels: Lista de níveis de confiança (ex: [0.95, 0.99])
        config: Configuração
    
    Returns:
        Dict com var_XX e cvar_XX para cada nível
    """
    
    if confidence_levels is None:
        confidence_levels = config.var_confidence_levels
    
    returns = np.diff(equity_curve) / equity_curve[:-1]
    
    if len(returns) < 10:
        # Retorna zeros se não houver dados suficientes
        result = {}
        for conf in confidence_levels:
            conf_pct = int(conf * 100)
            result[f'var_{conf_pct}'] = 0.0
            result[f'cvar_{conf_pct}'] = 0.0
        return result
    
    result = {}
    
    for conf in confidence_levels:
        # VaR: Percentil de perdas
        var = np.percentile(returns, (1 - conf) * 100)
        
        # CVaR: Média das perdas piores que VaR
        tail_losses = returns[returns <= var]
        cvar = np.mean(tail_losses) if len(tail_losses) > 0 else var
        
        conf_pct = int(conf * 100)
        result[f'var_{conf_pct}'] = float(var)
        result[f'cvar_{conf_pct}'] = float(cvar)
    
    return result


# =============================================================================
# FUNÇÕES DE CÁLCULO - TRADING METRICS
# =============================================================================

def calculate_trading_metrics(trades_df: pd.DataFrame) -> Dict[str, float]:
    """
    Calcula métricas de trading
    
    Args:
        trades_df: DataFrame com trades (deve ter coluna 'pnl_money' no mínimo)
    
    Returns:
        Dict com win_rate, profit_factor, expectancy, etc
    """
    
    if trades_df.empty:
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'expectancy': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'avg_r_multiple': 0.0,
            'largest_win': 0.0,
            'largest_loss': 0.0
        }
    
    total_trades = len(trades_df)
    
    # Separa vencedores e perdedores
    wins = trades_df[trades_df['pnl_money'] > 0]['pnl_money']
    losses = trades_df[trades_df['pnl_money'] < 0]['pnl_money']
    
    winning_trades = len(wins)
    losing_trades = len(losses)
    
    # Win Rate
    win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
    
    # Profit Factor
    gross_profit = wins.sum() if len(wins) > 0 else 0.0
    gross_loss = abs(losses.sum()) if len(losses) > 0 else 0.0
    
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0:
        profit_factor = float('inf')
    else:
        profit_factor = 0.0
    
    # Limita profit factor para serialização
    profit_factor = min(profit_factor, 999.0)
    
    # Médias
    avg_win = wins.mean() if len(wins) > 0 else 0.0
    avg_loss = losses.mean() if len(losses) > 0 else 0.0
    
    # Expectancy
    expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
    
    # R-Multiple médio
    if avg_loss != 0:
        avg_r_multiple = abs(avg_win / avg_loss)
    else:
        avg_r_multiple = 0.0
    
    # Largest
    largest_win = wins.max() if len(wins) > 0 else 0.0
    largest_loss = losses.min() if len(losses) > 0 else 0.0
    
    return {
        'total_trades': int(total_trades),
        'winning_trades': int(winning_trades),
        'losing_trades': int(losing_trades),
        'win_rate': float(win_rate),
        'profit_factor': float(profit_factor),
        'expectancy': float(expectancy),
        'avg_win': float(avg_win),
        'avg_loss': float(avg_loss),
        'avg_r_multiple': float(avg_r_multiple),
        'largest_win': float(largest_win),
        'largest_loss': float(largest_loss)
    }


# =============================================================================
# FUNÇÕES DE CÁLCULO - STREAKS
# =============================================================================

def calculate_streaks(trades_df: pd.DataFrame) -> Dict[str, Union[int, str]]:
    """
    Calcula sequências de vitórias/derrotas consecutivas
    
    Args:
        trades_df: DataFrame com trades (coluna 'pnl_money')
    
    Returns:
        Dict com max_consecutive_wins, max_consecutive_losses, current_streak
    """
    
    if trades_df.empty:
        return {
            'max_consecutive_wins': 0,
            'max_consecutive_losses': 0,
            'current_streak': 0,
            'current_streak_type': 'NONE'
        }
    
    # Cria série binária (1 = win, 0 = loss)
    wins = (trades_df['pnl_money'] > 0).astype(int)
    
    # Identifica mudanças de estado
    changes = wins.diff().fillna(wins.iloc[0])
    streak_ids = (changes != 0).cumsum()
    
    # Agrupa por streak
    streaks = trades_df.groupby(streak_ids).agg({
        'pnl_money': ['sum', 'count']
    })
    
    streaks.columns = ['sum', 'count']
    
    # Separa win e loss streaks
    win_streaks = streaks[streaks['sum'] > 0]['count']
    loss_streaks = streaks[streaks['sum'] < 0]['count']
    
    max_consecutive_wins = int(win_streaks.max()) if len(win_streaks) > 0 else 0
    max_consecutive_losses = int(loss_streaks.max()) if len(loss_streaks) > 0 else 0
    
    # Current streak
    last_pnl = trades_df['pnl_money'].iloc[-1]
    current_type = 'WIN' if last_pnl > 0 else 'LOSS' if last_pnl < 0 else 'NONE'
    
    # Conta streak atual
    current_streak = 1
    for i in range(len(trades_df) - 2, -1, -1):
        if (current_type == 'WIN' and trades_df['pnl_money'].iloc[i] > 0) or \
           (current_type == 'LOSS' and trades_df['pnl_money'].iloc[i] < 0):
            current_streak += 1
        else:
            break
    
    return {
        'max_consecutive_wins': max_consecutive_wins,
        'max_consecutive_losses': max_consecutive_losses,
        'current_streak': int(current_streak),
        'current_streak_type': current_type
    }


# =============================================================================
# FUNÇÕES DE CÁLCULO - EXPOSURE & EFFICIENCY
# =============================================================================

def calculate_exposure_metrics(
    trades_df: pd.DataFrame,
    total_bars: int,
    config: MetricsConfig = DEFAULT_CONFIG
) -> Dict[str, float]:
    """
    Calcula métricas de exposição ao mercado
    
    Args:
        trades_df: DataFrame com trades (deve ter 'entry_time' e 'exit_time')
        total_bars: Total de barras no período
        config: Configuração
    
    Returns:
        Dict com exposure_pct, avg_trade_duration, trades_per_day
    """
    
    if trades_df.empty or total_bars == 0:
        return {
            'exposure_pct': 0.0,
            'avg_trade_duration_hours': 0.0,
            'trades_per_day': 0.0
        }
    
    # Se não tiver timestamps, estima
    if 'entry_time' not in trades_df.columns or 'exit_time' not in trades_df.columns:
        # Assume duração média de 5 barras por trade
        avg_duration_bars = 5
        total_bars_in_trades = len(trades_df) * avg_duration_bars
        
        exposure_pct = min((total_bars_in_trades / total_bars) * 100, 100.0)
        avg_trade_duration_hours = (avg_duration_bars * 15) / 60  # M15 = 15 min
        
    else:
        # Calcula duração real de cada trade
        trades_df['duration'] = pd.to_datetime(trades_df['exit_time']) - pd.to_datetime(trades_df['entry_time'])
        avg_duration = trades_df['duration'].mean()
        
        # Converte para horas
        avg_trade_duration_hours = avg_duration.total_seconds() / 3600
        
        # Calcula exposição
        total_duration = trades_df['duration'].sum()
        total_time = total_bars * 15 * 60  # segundos
        
        exposure_pct = (total_duration.total_seconds() / total_time) * 100
    
    # Trades por dia
    total_days = total_bars / config.bars_per_day
    trades_per_day = len(trades_df) / total_days if total_days > 0 else 0.0
    
    return {
        'exposure_pct': float(min(exposure_pct, 100.0)),
        'avg_trade_duration_hours': float(avg_trade_duration_hours),
        'trades_per_day': float(trades_per_day)
    }


# =============================================================================
# FUNÇÃO PRINCIPAL - CALCULA TODAS AS MÉTRICAS
# =============================================================================

def calculate_all_metrics(
    equity_curve: np.ndarray,
    trades_df: pd.DataFrame,
    config: MetricsConfig = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> PerformanceMetrics:
    """
    🎯 FUNÇÃO PRINCIPAL: Calcula todas as métricas de uma vez
    
    Args:
        equity_curve: Array NumPy com curva de equity
        trades_df: DataFrame com trades (colunas mínimas: 'pnl_money')
        config: Configuração (usa DEFAULT_CONFIG se None)
        start_date: Data de início (string YYYY-MM-DD)
        end_date: Data de fim (string YYYY-MM-DD)
    
    Returns:
        PerformanceMetrics object completo
    
    Example:
        >>> equity = np.array([100000, 101000, 102500, ...])
        >>> trades = pd.DataFrame({'pnl_money': [500, -200, 1500, ...]})
        >>> metrics = calculate_all_metrics(equity, trades)
        >>> print(metrics)
        >>> print(f"Sharpe: {metrics.sharpe_ratio:.2f}")
    """
    
    if config is None:
        config = DEFAULT_CONFIG
    
    # Validações
    if len(equity_curve) < 2:
        logger.error("Equity curve deve ter pelo menos 2 pontos")
        raise ValueError("Insufficient data")
    
    if trades_df.empty:
        logger.warning("DataFrame de trades vazio")
    
    # Calcula cada grupo de métricas
    returns_metrics = calculate_returns(equity_curve, config)
    dd_metrics = calculate_drawdown_metrics(equity_curve, config)
    var_metrics = calculate_var_cvar(equity_curve, config=config)
    trading_metrics = calculate_trading_metrics(trades_df)
    streak_metrics = calculate_streaks(trades_df)
    exposure_metrics = calculate_exposure_metrics(trades_df, len(equity_curve), config)
    
    # Ratios ajustados ao risco
    sharpe = calculate_sharpe_ratio(equity_curve, config.risk_free_rate, config)
    sortino = calculate_sortino_ratio(equity_curve, config.risk_free_rate, config)
    calmar = calculate_calmar_ratio(equity_curve, config)
    omega = calculate_omega_ratio(equity_curve, config=config)
    ulcer = calculate_ulcer_index(equity_curve, config)
    
    # Volatilidade anualizada
    returns = np.diff(equity_curve) / equity_curve[:-1]
    volatility_annual = float(np.std(returns, ddof=1) * np.sqrt(config.bars_per_year))
    
    # Recovery Factor
    if dd_metrics['max_drawdown'] > 0:
        recovery_factor = returns_metrics['total_return'] / dd_metrics['max_drawdown']
    else:
        recovery_factor = float('inf') if returns_metrics['total_return'] > 0 else 0.0
    
    recovery_factor = min(recovery_factor, 999.0)  # Limita para serialização
    
    # Datas e período
    if start_date is None and 'entry_time' in trades_df.columns:
        start_date = str(pd.to_datetime(trades_df['entry_time'].min()).date())
    
    if end_date is None and 'exit_time' in trades_df.columns:
        end_date = str(pd.to_datetime(trades_df['exit_time'].max()).date())
    
    days_in_market = None
    if start_date and end_date:
        try:
            days_in_market = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
        except:
            pass
    
    # MAE/MFE (se disponível)
    avg_mae = None
    avg_mfe = None
    mae_to_profit_ratio = None
    
    if 'mae' in trades_df.columns and 'mfe' in trades_df.columns:
        avg_mae = float(trades_df['mae'].mean())
        avg_mfe = float(trades_df['mfe'].mean())
        
        if trading_metrics['avg_win'] > 0:
            mae_to_profit_ratio = abs(avg_mae / trading_metrics['avg_win'])
    
    # Monta objeto final
    metrics = PerformanceMetrics(
        # Retornos
        total_return=returns_metrics['total_return'],
        annualized_return=returns_metrics['annualized_return'],
        monthly_return=returns_metrics['monthly_return'],
        cagr=returns_metrics['cagr'],
        
        # Risco
        volatility_annual=volatility_annual,
        max_drawdown=dd_metrics['max_drawdown'],
        avg_drawdown=dd_metrics['avg_drawdown'],
        max_dd_duration_days=dd_metrics['max_dd_duration_days'],
        current_drawdown=dd_metrics['current_drawdown'],
        
        # Ratios
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        omega_ratio=omega,
        ulcer_index=ulcer,
        
        # Trading
        total_trades=trading_metrics['total_trades'],
        winning_trades=trading_metrics['winning_trades'],
        losing_trades=trading_metrics['losing_trades'],
        win_rate=trading_metrics['win_rate'],
        profit_factor=trading_metrics['profit_factor'],
        expectancy=trading_metrics['expectancy'],
        avg_win=trading_metrics['avg_win'],
        avg_loss=trading_metrics['avg_loss'],
        avg_r_multiple=trading_metrics['avg_r_multiple'],
        largest_win=trading_metrics['largest_win'],
        largest_loss=trading_metrics['largest_loss'],
        
        # Streaks
        max_consecutive_wins=streak_metrics['max_consecutive_wins'],
        max_consecutive_losses=streak_metrics['max_consecutive_losses'],
        current_streak=streak_metrics['current_streak'],
        current_streak_type=streak_metrics['current_streak_type'],
        
        # Exposure
        exposure_pct=exposure_metrics['exposure_pct'],
        avg_trade_duration_hours=exposure_metrics['avg_trade_duration_hours'],
        trades_per_day=exposure_metrics['trades_per_day'],
        
        # Risk
        var_95=var_metrics['var_95'],
        cvar_95=var_metrics['cvar_95'],
        var_99=var_metrics.get('var_99'),
        cvar_99=var_metrics.get('cvar_99'),
        
        # Recovery
        recovery_factor=recovery_factor,
        recovery_time_days=dd_metrics['recovery_time_days'],
        
        # MAE/MFE
        avg_mae=avg_mae,
        avg_mfe=avg_mfe,
        mae_to_profit_ratio=mae_to_profit_ratio,
        
        # Timing
        start_date=start_date,
        end_date=end_date,
        days_in_market=days_in_market,
        
        # Equity
        initial_equity=float(equity_curve[0]),
        final_equity=float(equity_curve[-1]),
        peak_equity=float(np.max(equity_curve))
    )
    
    return metrics


# =============================================================================
# UTILITÁRIOS
# =============================================================================

def compare_strategies(metrics_list: List[PerformanceMetrics]) -> pd.DataFrame:
    """
    Compara múltiplas estratégias lado a lado
    
    Args:
        metrics_list: Lista de PerformanceMetrics
    
    Returns:
        DataFrame com comparação
    """
    
    data = []
    for i, m in enumerate(metrics_list, 1):
        data.append({
            'Strategy': f'Strategy {i}',
            'Total Return': f"{m.total_return:.2%}",
            'Sharpe': f"{m.sharpe_ratio:.2f}",
            'Sortino': f"{m.sortino_ratio:.2f}",
            'Calmar': f"{m.calmar_ratio:.2f}",
            'Max DD': f"{m.max_drawdown:.2%}",
            'Win Rate': f"{m.win_rate:.1%}",
            'Profit Factor': f"{m.profit_factor:.2f}",
            'Trades': m.total_trades,
            'Grade': m.get_grade()
        })
    
    return pd.DataFrame(data)


def export_metrics_to_json(metrics: PerformanceMetrics, filepath: str):
    """Exporta métricas para arquivo JSON"""
    import json
    
    with open(filepath, 'w') as f:
        json.dump(metrics.to_dict(), f, indent=2, default=str)
    
    logger.info(f"Métricas exportadas para {filepath}")


def load_metrics_from_json(filepath: str) -> PerformanceMetrics:
    """Carrega métricas de arquivo JSON"""
    import json
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    return PerformanceMetrics(**data)


# =============================================================================
# EXEMPLO DE USO
# =============================================================================

if __name__ == "__main__":
    """
    Exemplo de uso do módulo
    """
    
    # Simula dados
    np.random.seed(42)
    
    # Equity curve simulada (random walk com drift positivo)
    n_bars = 5000
    returns = np.random.normal(0.0005, 0.01, n_bars)
    equity = 100000 * np.cumprod(1 + returns)
    
    # Trades simulados
    n_trades = 150
    trades_data = {
        'pnl_money': np.random.normal(500, 1000, n_trades),
        'entry_time': pd.date_range('2025-01-01', periods=n_trades, freq='H'),
        'exit_time': pd.date_range('2025-01-01', periods=n_trades, freq='H') + pd.Timedelta(hours=3)
    }
    trades_df = pd.DataFrame(trades_data)
    
    # Calcula métricas
    print("Calculando métricas...")
    metrics = calculate_all_metrics(
        equity_curve=equity,
        trades_df=trades_df,
        start_date='2025-01-01',
        end_date='2025-12-26'
    )
    
    # Exibe
    print(metrics)
    print(f"\n📊 Nota Final: {metrics.get_grade()}")
    
    # Exporta
    export_metrics_to_json(metrics, 'example_metrics.json')