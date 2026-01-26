import os

# ===========================
# PARÂMETROS DO PROJETO XP3/B3
# ===========================

# Limites gerais do bot
MAX_SYMBOLS = 10
MAX_PER_SECTOR = 2

# Mapa de setores (todos os ativos monitorados)
SECTOR_MAP = {
    "ITUB4": "FINANCIALS", "PETR4": "ENERGY", "VALE3": "MATERIALS", "BPAC11": "FINANCIALS",
    "ABEV3": "CONSUMER_STAPLES", "BBDC4": "FINANCIALS", "WEGE3": "INDUSTRIALS", "RDOR3": "HEALTHCARE",
    "BBAS3": "FINANCIALS", "SANB11": "FINANCIALS", "SUZB3": "MATERIALS", "B3SA3": "FINANCIALS",
    "GGBR4": "MATERIALS", "CSNA3": "MATERIALS", "USIM5": "MATERIALS", "JBSS3": "CONSUMER_STAPLES",
    "RAIL3": "INDUSTRIALS", "VIVT3": "COMMUNICATIONS", "AZZA3": "MATERIALS", "PRIO3": "ENERGY",
    "RADL3": "HEALTHCARE", "LREN3": "CONSUMER_DISCRETIONARY", "MGLU3": "CONSUMER_DISCRETIONARY",
    "ELET3": "UTILITIES", "TAEE11": "UTILITIES", "ENGI11": "UTILITIES", "CPLE6": "UTILITIES",
    "CMIG4": "UTILITIES", "ITSA4": "FINANCIALS", "BBSE3": "FINANCIALS", "CRFB3": "CONSUMER_STAPLES",
    "ASAI3": "CONSUMER_STAPLES", "HYPE3": "HEALTHCARE", "TOTS3": "TECHNOLOGY", "VIVA3": "CONSUMER_DISCRETIONARY",
    "NTCO3": "CONSUMER_STAPLES", "BRFS3": "CONSUMER_STAPLES", "CCRO3": "INDUSTRIALS", "ECOR3": "INDUSTRIALS",
    "RENT3": "CONSUMER_DISCRETIONARY", "AZUL4": "INDUSTRIALS", "BEEF3": "CONSUMER_STAPLES", "QUAL3": "HEALTHCARE",
    "TRPL4": "UTILITIES", "SAPR11": "UTILITIES", "CSMG3": "UTILITIES", "SBSP3": "UTILITIES",
    "FLRY3": "HEALTHCARE", "EVEN3": "REAL_ESTATE", "MDIA3": "CONSUMER_STAPLES", "CYRE3": "REAL_ESTATE",
    "BRKM5": "MATERIALS", "KLBN11": "MATERIALS", "MRFG3": "CONSUMER_STAPLES", "COGN3": "CONSUMER_DISCRETIONARY",
    "ANIM3": "CONSUMER_DISCRETIONARY", "SEER3": "CONSUMER_DISCRETIONARY", "YDUQ3": "CONSUMER_DISCRETIONARY",
    "AURE3": "FINANCIALS", "VAMO3": "INDUSTRIALS", "VBBR3": "ENERGY", "RECV3": "ENERGY",
    "CSAN3": "ENERGY", "POMO4": "FINANCIALS", "DIRR3": "REAL_ESTATE", "ENEV3": "ENERGY",
    "EQTL3": "FINANCIALS", "MULT3": "REAL_ESTATE", "VISC3": "CONSUMER_DISCRETIONARY", "ALUP11": "UTILITIES",
    "FESA4": "MATERIALS", "KEPL3": "INDUSTRIALS", "ROMI3": "INDUSTRIALS", "AERI3": "TECHNOLOGY",
    "CEAB3": "CONSUMER_DISCRETIONARY", "SRNA3": "UTILITIES", "HBOR3": "REAL_ESTATE", "MDNE3": "REAL_ESTATE",
    "CASH3": "TECHNOLOGY", "AURA33": "MATERIALS", "SOMA3": "CONSUMER_DISCRETIONARY", "NICE3": "TECHNOLOGY",
    "LWSA3": "TECHNOLOGY", "BEES3": "FINANCIALS", "IFCM3": "TECHNOLOGY", "PMAM3": "CONSUMER_STAPLES",
    "AGXY3": "FINANCIALS", "BLAU3": "HEALTHCARE", "BMGB4": "FINANCIALS", "BRIT3": "CONSUMER_DISCRETIONARY",
    "CLSA3": "CONSUMER_DISCRETIONARY", "CSED3": "INDUSTRIALS", "DXCO3": "MATERIALS", "EALT3": "REAL_ESTATE",
    "ELMD3": "COMMUNICATIONS", "ENJU3": "CONSUMER_DISCRETIONARY", "FFOR3": "FINANCIALS", "GOLL4": "INDUSTRIALS",
    "GRND3": "REAL_ESTATE"
}

# Lista de símbolos proxy (usada em alguns módulos antigos - pode manter)
PROXY_SYMBOLS = [
    "VALE3", "PETR4", "ITUB4", "BBDC4", "BBAS3",
    "ABEV3", "WEGE3", "JBSS3", "RENT3", "PRIO3",
    "SUZB3", "ELET3", "VIVT3", "HAPV3"
]

# ===========================
# HORÁRIOS DE OPERAÇÃO
# ===========================
TRADING_START = "10:20"
TRADING_END   = "16:40"
FRIDAY_REDUCED_RISK_AFTER = "15:00"
NO_ENTRY_AFTER = "16:20"           # Não abre novas posições após esse horário
DAILY_RESET_TIME = "10:15"         # Reset diário do circuit breaker

# ===========================
# GESTÃO DE RISCO
# ===========================
RISK_PER_TRADE_PCT = 0.01          # 1% da equity por trade padrão
REDUCED_RISK_PCT = 0.005           # 0.5% na sexta após 15h
MAX_DAILY_DRAWDOWN_PCT = 0.02       # Circuit breaker diário (2%)

MAX_RISK_PER_SYMBOL_PCT = 0.04     # Máximo 4% da equity por papel
SYMBOL_BLOCK_LOSS_PCT = 0.025      # Bloqueia ativo após perda de 2.5%
SYMBOL_BLOCK_HOURS = 72

MAX_TRADE_DURATION_CANDLES = 48    # Time-stop

# ===========================
# PYRAMIDING
# ===========================
ENABLE_PYRAMID = True
PYRAMID_MAX_LEGS = 2
PYRAMID_ATR_DISTANCE = 1.0         # Segunda perna só após +1.0 ATR a favor
PYRAMID_RISK_SPLIT = [0.6, 0.4]     # 60% na primeira, 40% na segunda

# ===========================
# STOP LOSS / TAKE PROFIT
# ===========================
SL_ATR_MULTIPLIER = 2.0            # SL inicial = preço ± ATR × 2.0
TP_ATR_MULT = 3.0                  # TP opcional (não usado atualmente, mas disponível)
TRAILING_STEP_ATR_MULTIPLIER = 1.0

# ===========================
# FILTROS PROFISSIONAIS B3
# ===========================
MIN_AVG_VOLUME_20 = 500000         # Volume médio 20 períodos mínimo
MAX_GAP_OPEN_PCT = 0.03            # Gap de abertura > 3% → bloqueia entrada

VOLATILITY_MIN_MULT = 0.60
VOLATILITY_MAX_MULT = 2.50

# ===========================
# FILTRO DE CORRELAÇÃO
# ===========================
ENABLE_CORRELATION_FILTER = True
MIN_CORRELATION_SCORE_TO_BLOCK = 0.70
CORRELATION_LOOKBACK_DAYS = 60

# ===========================
# FILTRO MACRO
# ===========================
MACRO_TIMEFRAME = "H1"
MACRO_EMA_LONG = 200

# ===========================
# MODOS E CONTROLES
# ===========================
TRADE_BOTH_DIRECTIONS = True
FAST_LOOP_INTERVAL_SECONDS = 1.0

# ===========================
# OTIMIZADOR
# ===========================
WFO_OOS_RATIO = 0.30
ENABLE_MONTE_CARLO = True

OPTIMIZER_OUTPUT = "optimizer_output"
OPTIMIZER_HISTORY_FILE = os.path.join(OPTIMIZER_OUTPUT, "history.json")

# ===========================
# PARÂMETROS OTIMIZADOS MANUAIS (ELITE)
# ===========================

# ===========================
# PARÂMETROS OTIMIZADOS MANUAIS (ELITE)
# ===========================

# ===========================
# PARÂMETROS OTIMIZADOS MANUAIS (ELITE)
# ===========================
ELITE_SYMBOLS = {
    "B3SA3": {'ema_short': 5, 'ema_long': 111, 'rsi_low': 23, 'rsi_high': 71, 'adx_threshold': 17, 'mom_min': 0.08000000000000002},
    "MGLU3": {'ema_short': 17, 'ema_long': 41, 'rsi_low': 33, 'rsi_high': 76, 'adx_threshold': 35, 'mom_min': 0.07},
    "SBSP3": {'ema_short': 34, 'ema_long': 47, 'rsi_low': 27, 'rsi_high': 56, 'adx_threshold': 26, 'mom_min': 0.08000000000000002},
    "MDIA3": {'ema_short': 6, 'ema_long': 52, 'rsi_low': 40, 'rsi_high': 59, 'adx_threshold': 18, 'mom_min': 0.12000000000000002},
    "YDUQ3": {'ema_short': 35, 'ema_long': 45, 'rsi_low': 27, 'rsi_high': 56, 'adx_threshold': 29, 'mom_min': 0.15},
    "AURE3": {'ema_short': 20, 'ema_long': 52, 'rsi_low': 37, 'rsi_high': 76, 'adx_threshold': 35, 'mom_min': 0.09},
    "RECV3": {'ema_short': 12, 'ema_long': 52, 'rsi_low': 34, 'rsi_high': 57, 'adx_threshold': 16, 'mom_min': 0.11000000000000001},
    "CSAN3": {'ema_short': 34, 'ema_long': 41, 'rsi_low': 33, 'rsi_high': 79, 'adx_threshold': 29, 'mom_min': 0.0},
    "ENEV3": {'ema_short': 5, 'ema_long': 43, 'rsi_low': 27, 'rsi_high': 61, 'adx_threshold': 33, 'mom_min': 0.09},
    "CEAB3": {'ema_short': 5, 'ema_long': 55, 'rsi_low': 20, 'rsi_high': 64, 'adx_threshold': 39, 'mom_min': 0.05000000000000002},
    "CSED3": {'ema_short': 29, 'ema_long': 40, 'rsi_low': 31, 'rsi_high': 67, 'adx_threshold': 20, 'mom_min': 0.1},
    "DXCO3": {'ema_short': 5, 'ema_long': 95, 'rsi_low': 36, 'rsi_high': 71, 'adx_threshold': 26, 'mom_min': 0.08000000000000002},
}
