"""
CONFIGURAÇÃO DE RISCO XP3 v5

Centraliza parâmetros de gestão de risco, tuning e sanity checks.
"""

# A.1 Dynamic Position Sizing
DYNAMIC_POSITION_SIZING = {
    "atr_high_pct": 0.04,          # Reduz 50% se ATR% > 4%
    "atr_mid_pct": 0.025,          # Reduz 25% se ATR% > 2.5%
    "beta_threshold": 1.30,         # Reduz 30% se Beta > 1.3
    "dd_threshold": 0.15,           # Reduz 50% se DD > 15%
}

# A.2 Circuit Breakers
CIRCUIT_BREAKERS = {
    "max_consecutive_losses": 3,    # Pausa ao atingir 3 perdas seguidas
    "pause_bars_m15": 200,          # Aproximadamente 2 dias em M15
    "intraday_dd_limit": 0.10,      # Pausa se DD intraday > 10%
}

# A.3 RR Assimétrico
ASYMMETRIC_RR = {
    "wr_low": 0.40,                 # TP reduzido se WR < 40%
    "wr_high": 0.60,                # TP ampliado se WR > 60%
    "short_tp_factor": 0.90,        # Shorts mais conservadores (TP 90%)
    "short_risk_factor": 0.80,      # Menor tamanho em shorts (80%)
}

# C. Optuna Tuning
OPTUNA_TUNING = {
    "ema_short_min": 8,
    "ema_short_max": 30,
    "ema_long_min": 35,
    "ema_long_max": 100,
    "rsi_low_min": 25,
    "rsi_low_max": 40,
    "rsi_high_min": 60,
    "rsi_high_max": 80,
    "adx_min": 15,
    "adx_max": 35,
    "sl_mult_min": 1.5,
    "sl_mult_max": 3.5,
    "sl_mult_step": 0.1,
    "tp_ratio_min": 1.2,
    "tp_ratio_max": 3.0,
    "tp_ratio_step": 0.2,
    "base_slippage": 0.0015,
    "n_trials": 150,
    "timeout_sec": 1500,
}

# F. Markowitz Protegido
MARKOWITZ_RULES = {
    "sector_cap": 0.25,             # Teto por setor
    "blue_min": 0.50,               # Mínimo Blue Chips
    "opp_max": 0.50,                # Máximo Oportunidades
    "prefilter_dd_max": 0.65,       # Exclui DD >= 65%
    "prefilter_trades_min": 10,     # Exclui < 10 trades
    "prefilter_liquidity_min": 10_000_000,  # Liquidez mínima
}

# I. Sanity Checks
SANITY_CHECKS = {
    "min_wr_forward": 0.30,
    "min_calmar_forward": 0.0,
    "min_calmar_stress": -0.20,
    "min_ratio_vs_buyhold": 0.50,
}

