import os

# ===========================
# PARÂMETROS DO PROJETO XP3/B3
# ===========================

# Limites do bot
MAX_SYMBOLS = 10
MAX_PER_SECTOR = 2

# Timeframe usado no bot_fast e no optimizer
TIMEFRAME_DEFAULT = "M15"   # M15 (pode usar "H1" se preferir)

# Setores dos ativos (mantive seu mapa original)
SECTOR_MAP = {
    "ITUB4": "FINANCIALS",
    "PETR4": "ENERGY",
    "VALE3": "MATERIALS",
    "BPAC11": "FINANCIALS",
    "ABEV3": "CONSUMER_STAPLES",
    "BBDC4": "FINANCIALS",
    "WEGE3": "INDUSTRIALS",
    "RDOR3": "HEALTHCARE",
    "BBAS3": "FINANCIALS",
    "SANB11": "FINANCIALS",
    "SUZB3": "MATERIALS",
    "B3SA3": "FINANCIALS",
    "GGBR4": "MATERIALS",
    "CSNA3": "MATERIALS",
    "USIM5": "MATERIALS",
    "JBSS3": "CONSUMER_STAPLES",
    "RAIL3": "INDUSTRIALS",
    "VIVT3": "COMMUNICATIONS",
    "AZZA3": "MATERIALS",
    "PRIO3": "ENERGY",
    "RADL3": "HEALTHCARE",
    "LREN3": "CONSUMER_DISCRETIONARY",
    "MGLU3": "CONSUMER_DISCRETIONARY",
    "ELET3": "UTILITIES",
    "TAEE11": "UTILITIES",
    "ENGI11": "UTILITIES",
    "CPLE6": "UTILITIES",
    "CMIG4": "UTILITIES",
    "ITSA4": "FINANCIALS",
    "BBSE3": "FINANCIALS",
    "CRFB3": "CONSUMER_STAPLES",
    "ASAI3": "CONSUMER_STAPLES",
    "HYPE3": "HEALTHCARE",
    "TOTS3": "TECHNOLOGY",
    "VIVA3": "CONSUMER_DISCRETIONARY",
    "NTCO3": "CONSUMER_STAPLES",
    "BRFS3": "CONSUMER_STAPLES",
    "CCRO3": "INDUSTRIALS",
    "ECOR3": "INDUSTRIALS",
    "RENT3": "CONSUMER_DISCRETIONARY",
    "AZUL4": "INDUSTRIALS",
    "BEEF3": "CONSUMER_STAPLES",
    "QUAL3": "HEALTHCARE",
    "TRPL4": "UTILITIES",
    "SAPR11": "UTILITIES",
    "CSMG3": "UTILITIES",
    "SBSP3": "UTILITIES",
    "FLRY3": "HEALTHCARE",
    "EVEN3": "REAL_ESTATE",
    "MDIA3": "CONSUMER_STAPLES",
    "CYRE3": "REAL_ESTATE",
    "BRKM5": "MATERIALS",
    "KLBN11": "MATERIALS",
    "MRFG3": "CONSUMER_STAPLES",
    "COGN3": "CONSUMER_DISCRETIONARY",
    "ANIM3": "CONSUMER_DISCRETIONARY",
    "SEER3": "CONSUMER_DISCRETIONARY",
    "YDUQ3": "CONSUMER_DISCRETIONARY",
    "AURE3": "FINANCIALS",
    "VAMO3": "INDUSTRIALS",
    "VBBR3": "ENERGY",
    "RECV3": "ENERGY",
    "CSAN3": "ENERGY",
    "POMO4": "FINANCIALS",
    "DIRR3": "REAL_ESTATE",
    "ENEV3": "ENERGY",
    "EQTL3": "FINANCIALS",
    "MULT3": "REAL_ESTATE",
    "VISC3": "CONSUMER_DISCRETIONARY",
    "ALUP11": "UTILITIES",
    "FESA4": "MATERIALS",
    "KEPL3": "INDUSTRIALS",
    "ROMI3": "INDUSTRIALS",
    "AERI3": "TECHNOLOGY",
    "CEAB3": "CONSUMER_DISCRETIONARY",
    "SRNA3": "UTILITIES",
    "HBOR3": "REAL_ESTATE",
    "MDNE3": "REAL_ESTATE",
    "CASH3": "TECHNOLOGY",
    "AURA33": "MATERIALS",
    "SOMA3": "CONSUMER_DISCRETIONARY",
    "NICE3": "TECHNOLOGY",
    "LWSA3": "TECHNOLOGY",
    "BEES3": "FINANCIALS",
    "IFCM3": "TECHNOLOGY",
    "PMAM3": "CONSUMER_STAPLES",
    "AGXY3": "FINANCIALS",
    "BLAU3": "HEALTHCARE",
    "BMGB4": "FINANCIALS",
    "BRIT3": "CONSUMER_DISCRETIONARY",
    "CLSA3": "CONSUMER_DISCRETIONARY",
    "CSED3": "INDUSTRIALS",
    "DXCO3": "MATERIALS",
    "EALT3": "REAL_ESTATE",
    "ELMD3": "COMMUNICATIONS",
    "ENJU3": "CONSUMER_DISCRETIONARY",
    "FFOR3": "FINANCIALS",
    "GOLL4": "INDUSTRIALS",
    "GRND3": "REAL_ESTATE"
}

# Lista principal usada pelo bot
PROXY_SYMBOLS = [
    "VALE3", "PETR4", "ITUB4", "BBDC4", "BBAS3",
    "ABEV3", "WEGE3", "JBSS3", "RENT3", "PRIO3",
    "SUZB3", "ELET3", "VIVT3", "HAPV3"
]

# ===========================
# PARÂMETROS DO OPTIMIZER TURBO
# ===========================

WFO_IN_SAMPLE_DAYS = 200
WFO_OOS_DAYS = 50
WFO_WINDOWS = 6

GRID = {
    "ema_short": [5, 8, 9, 12],
    "ema_long": [20, 26, 30],
    "rsi_period": [7, 14],
}

DEFAULT_PARAMS = {
    "ema_short": 9,
    "ema_long": 21,
    "rsi_period": 14,
    "mom_period": 10,
    "adx_period": 14,
    "rsi_overbought": 70,
    "rsi_oversold": 30,
    "adx_threshold": 20,
}

OPTIMIZER_OUTPUT = "optimizer_output"
OPTIMIZER_HISTORY_FILE = os.path.join(OPTIMIZER_OUTPUT, "history.json")

# ===========================
# PARÂMETROS DE RISCO
# ===========================
MAX_TOTAL_DRAWDOWN_PCT = 0.03
VAR_95_DAILY_LIMIT = 0.02
CLOSE_POSITIONS_ON_CB = True
MIN_ADV_20D_BRL = 2_000_000

# Circuit Breaker financeiro (em BRL)
MAX_DAILY_LOSS_BRL = 15000.0  # <-- default R$ 15.000,00

# Se true, ao disparar circuit-breaker o bot fecha posições abertas automaticamente.
CB_CLOSE_POSITIONS = True

# ===========================
# CONTROLES DO BOT
# ===========================
DUP_ORDER_COOLDOWN_SECONDS = 60
SCAN_INTERVAL_SECONDS = 60
FAST_LOOP_INTERVAL_SECONDS = 1.0  # protege o loop rápido para 1s
MARKET_OPEN_HOUR = 10
MARKET_OPEN_MINUTE = 15
MARKET_CLOSE_HOUR = 16
MARKET_CLOSE_MINUTE = 50
MAX_TRADES_PER_CYCLE = 3

# ===========================
# TRAILING STOP / ATR
# ===========================
# O SL só move se o preço se afastar TRAILING_STEP_ATR_MULTIPLIER * ATR
TRAILING_STEP_ATR_MULTIPLIER = 1.0

# multiplicador usado para calcular novo SL = price - ATR * SL_ATR_MULT (ou + para venda)
SL_ATR_MULT = 2.0
TP_ATR_MULT = 3.0

# ===========================
# BOT STATE (persistência)
# ===========================
BOT_STATE_FILE = "bot_state.json"

# ===========================
# Outros defaults (podem ser sobrescritos)
# ===========================
MAX_SYMBOLS = 10
MAX_PER_SECTOR = 2

PROXY_MODE = "AUTO"
PROXY_AUTO_N = 15
PROXY_AUTO_WORKERS = 12



OPTIMIZED_PARAMS = {
  "ABEV3": {
    "ema_short": 5,
    "ema_long": 20,
    "rsi_period": 7,
    "adx_period": 10,
    "adx_threshold": 20.0,
    "rsi_low": 40.0,
    "rsi_high": 60.0,
    "mom_min": 0.0
  },
  "B3SA3": {
    "ema_short": 5,
    "ema_long": 20,
    "rsi_period": 7,
    "adx_period": 20,
    "adx_threshold": 20.0,
    "rsi_low": 30.0,
    "rsi_high": 60.0,
    "mom_min": 0.0
  },
  "BBDC4": {
    "ema_short": 5,
    "ema_long": 20,
    "rsi_period": 7,
    "adx_period": 10,
    "adx_threshold": 20.0,
    "rsi_low": 30.0,
    "rsi_high": 70.0,
    "mom_min": 0.0
  },
  "BPAC11": {
    "ema_short": 5,
    "ema_long": 20,
    "rsi_period": 7,
    "adx_period": 20,
    "adx_threshold": 25.0,
    "rsi_low": 40.0,
    "rsi_high": 60.0,
    "mom_min": 0.0
  },
  "CEAB3": {
    "ema_short": 5,
    "ema_long": 20,
    "rsi_period": 7,
    "adx_period": 10,
    "adx_threshold": 30.0,
    "rsi_low": 30.0,
    "rsi_high": 70.0,
    "mom_min": 0.0
  },
  "CSAN3": {
    "ema_short": 5,
    "ema_long": 20,
    "rsi_period": 7,
    "adx_period": 14,
    "adx_threshold": 20.0,
    "rsi_low": 30.0,
    "rsi_high": 65.0,
    "mom_min": 0.0
  },
  "CSNA3": {
    "ema_short": 5,
    "ema_long": 20,
    "rsi_period": 7,
    "adx_period": 20,
    "adx_threshold": 30.0,
    "rsi_low": 30.0,
    "rsi_high": 60.0,
    "mom_min": 0.0
  },
  "DIRR3": {
    "ema_short": 5,
    "ema_long": 20,
    "rsi_period": 7,
    "adx_period": 10,
    "adx_threshold": 20.0,
    "rsi_low": 40.0,
    "rsi_high": 60.0,
    "mom_min": 0.0
  },
  "ITSA4": {
    "ema_short": 5,
    "ema_long": 20,
    "rsi_period": 7,
    "adx_period": 10,
    "adx_threshold": 30.0,
    "rsi_low": 30.0,
    "rsi_high": 70.0,
    "mom_min": 0.0
  },
  "ITUB4": {
    "ema_short": 5,
    "ema_long": 20,
    "rsi_period": 7,
    "adx_period": 10,
    "adx_threshold": 20.0,
    "rsi_low": 40.0,
    "rsi_high": 65.0,
    "mom_min": 0.0
  },
  "LREN3": {
    "ema_short": 5,
    "ema_long": 20,
    "rsi_period": 7,
    "adx_period": 14,
    "adx_threshold": 30.0,
    "rsi_low": 35.0,
    "rsi_high": 60.0,
    "mom_min": 0.0
  },
  "PETR4": {
    "ema_short": 5,
    "ema_long": 20,
    "rsi_period": 7,
    "adx_period": 14,
    "adx_threshold": 25.0,
    "rsi_low": 30.0,
    "rsi_high": 65.0,
    "mom_min": 0.0
  },
  "RENT3": {
    "ema_short": 5,
    "ema_long": 20,
    "rsi_period": 7,
    "adx_period": 20,
    "adx_threshold": 30.0,
    "rsi_low": 30.0,
    "rsi_high": 60.0,
    "mom_min": 0.0
  },
  "SUZB3": {
    "ema_short": 5,
    "ema_long": 20,
    "rsi_period": 7,
    "adx_period": 20,
    "adx_threshold": 30.0,
    "rsi_low": 30.0,
    "rsi_high": 60.0,
    "mom_min": 0.0
  },
  "VALE3": {
    "ema_short": 5,
    "ema_long": 20,
    "rsi_period": 7,
    "adx_period": 10,
    "adx_threshold": 15.0,
    "rsi_low": 30.0,
    "rsi_high": 70.0,
    "mom_min": 0.0
  }
}

# Risco por trade (percentual da equity) - usado pelo calculate_position_size
RISK_PER_TRADE_PCT = 0.01  # 1% por padrão


# Filtro macro (Multi-Timeframe)
MACRO_TIMEFRAME = "H1"     # ex: "H1" ou "D1"
MACRO_EMA_LONG = 200       # período da EMA longa no timeframe macro usada para confirmação de tendência


# Fraction of MAX_TOTAL_DRAWDOWN_PCT below which circuit breaker will auto-reset
CB_RESET_THRESHOLD = 0.7

# ===========================
# MODO BIDIRECIONAL & FILTROS PROFISSIONAIS
# ===========================
TRADE_BOTH_DIRECTIONS = True          # ← ATIVADO
ENABLE_PYRAMID = True                 # 2 pernas
PYRAMID_ATR_DISTANCE = 1.0            # segunda entrada após +1.0 ATR a favor
PYRAMID_RISK_SPLIT = [0.6, 0.4]       # 60% + 40%

# Filtros de volatilidade
VOLATILITY_MIN_MULT = 0.60
VOLATILITY_MAX_MULT = 2.50

# Horários proibidos (horário de Brasília)
TRADING_START = "10:20"
TRADING_END   = "16:40"
FRIDAY_REDUCED_RISK_AFTER = "15:00"
REDUCED_RISK_PCT = 0.005  # 0.5% na sexta após 15h
# Horário (B3) para reset diário do circuito breaker
DAILY_RESET_TIME = "10:15"      # após abertura oficial

# Circuit breaker por ativo
MAX_RISK_PER_SYMBOL_PCT = 0.04        # 4% da equity máximo por papel
SYMBOL_BLOCK_LOSS_PCT = 0.025         # 2.5% perda → bloqueia 72h
SYMBOL_BLOCK_HOURS = 72

# ===== SAFETY / LATENCY =====
MAX_MT5_LATENCY_MS = 400        # acima disso, pausa trading
LATENCY_SAMPLE_INTERVAL = 5     # segundos
AUTO_RESUME_LATENCY_MS = 250

# ===== FALLBACK =====
ENABLE_TIMEFRAME_FALLBACK = True
FALLBACK_TIMEFRAME = "M1"
BREAKOUT_ATR_MULT = 2.2

# ===== LOG =====
ENABLE_SLIPPAGE_LOG = True
MAX_DAILY_SLIPPAGE_ATR = 1.2

MAX_DAILY_DRAWDOWN_PCT = 0.02  # 2% máximo de drawdown diário
RISK_PER_TRADE_PCT = 0.01  # Risco padrão por trade
REDUCED_RISK_PCT = 0.005  # Risco reduzido na sexta-feira
FRIDAY_REDUCED_RISK_AFTER = "16:40"  # Após esse horário, risco é reduzido
TRADING_END = "16:40"
