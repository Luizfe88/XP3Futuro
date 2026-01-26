import os

# ===========================
# PARÂMETROS DO PROJETO XP3/B3
# ===========================

# Limites gerais do bot
MAX_SYMBOLS = 10
MAX_PER_SECTOR = 2
MAX_CORRELATION_PAIRS = 3

# Mapa de setores (todos os ativos monitorados)
SECTOR_MAP = {
    # --- FINANCEIRO E HOLDINGS (35) ---
    "ITUB4": "FINANCIALS", "BBDC4": "FINANCIALS", "BBAS3": "FINANCIALS", "SANB11": "FINANCIALS", 
    "ITSA4": "FINANCIALS", "BPAC11": "FINANCIALS", "B3SA3": "FINANCIALS", "BBSE3": "FINANCIALS", 
    "PSSA3": "FINANCIALS", "CXSE3": "FINANCIALS", "IRBR3": "FINANCIALS", "ABCB4": "FINANCIALS", 
    "BEES3": "FINANCIALS", "BMGB4": "FINANCIALS", "BBDC3": "FINANCIALS", "SANB3": "FINANCIALS", 
    "ITUB3": "FINANCIALS", "BRSR6": "FINANCIALS", "CIEL3": "FINANCIALS", "WIZC3": "FINANCIALS",
    "BIDI11": "FINANCIALS", "AMBP3": "FINANCIALS", "BNCP3": "FINANCIALS", "MODL3": "FINANCIALS",
    "AGXY3": "FINANCIALS", "BRML3": "FINANCIALS", "IGTI11": "FINANCIALS", "ALPK3": "FINANCIALS",
    "BNBR3": "FINANCIALS", "PINE4": "FINANCIALS", "BRGE11": "FINANCIALS", "BSLI4": "FINANCIALS",
    "BMEB4": "FINANCIALS", "BEES4": "FINANCIALS", "BAZA3": "FINANCIALS",

    # --- ENERGIA, PETRÓLEO E GÁS (15) ---
    "PETR4": "ENERGY", "PETR3": "ENERGY", "PRIO3": "ENERGY", "RECV3": "ENERGY", 
    "RRRP3": "ENERGY", "CSAN3": "ENERGY", "VBBR3": "ENERGY", "UGPA3": "ENERGY", 
    "ENAT3": "ENERGY", "RPMG3": "ENERGY", "OSXB3": "ENERGY", "DMMO3": "ENERGY",
    "RAIZ4": "ENERGY", "PETZ3": "ENERGY", "OPCT3": "ENERGY",

    # --- MATERIAIS BÁSICOS (Siderurgia, Papel, Mineração) (25) ---
    "VALE3": "MATERIALS", "SUZB3": "MATERIALS", "KLBN11": "MATERIALS", "GGBR4": "MATERIALS", 
    "GOAU4": "MATERIALS", "CSNA3": "MATERIALS", "USIM5": "MATERIALS", "BRKM5": "MATERIALS", 
    "UNIP6": "MATERIALS", "FESA4": "MATERIALS", "RANI3": "MATERIALS", "DXCO3": "MATERIALS", 
    "AURA33": "MATERIALS", "CBAV3": "MATERIALS", "PMAM3": "MATERIALS", "GGBR3": "MATERIALS",
    "CSAN3": "MATERIALS", "GOAU3": "MATERIALS", "USIM3": "MATERIALS", "KLBN3": "MATERIALS",
    "FHER3": "MATERIALS", "EUCA4": "MATERIALS", "DEXP3": "MATERIALS", "MNPR3": "MATERIALS",
    "CRIV4": "MATERIALS",

    # --- UTILIDADE PÚBLICA (Elétricas e Saneamento) (35) ---
    "ELET3": "UTILITIES", "ELET6": "UTILITIES", "CPLE6": "UTILITIES", "CMIG4": "UTILITIES", 
    "EQTL3": "UTILITIES", "SBSP3": "UTILITIES", "SAPR11": "UTILITIES", "CSMG3": "UTILITIES", 
    "TRPL4": "UTILITIES", "TAEE11": "UTILITIES", "ENGI11": "UTILITIES", "ALUP11": "UTILITIES", 
    "EGIE3": "UTILITIES", "CPFE3": "UTILITIES", "AURE3": "UTILITIES", "ENEV3": "UTILITIES", 
    "NEOE3": "UTILITIES", "AESB3": "UTILITIES", "STBP3": "UTILITIES", "LIGT3": "UTILITIES",
    "ALIANÇA": "UTILITIES", "CESP6": "UTILITIES", "CEEB3": "UTILITIES", "CPLE3": "UTILITIES",
    "CMIG3": "UTILITIES", "ENBR3": "UTILITIES", "GEPA4": "UTILITIES", "KEPL3": "UTILITIES",
    "RNEW11": "UTILITIES", "CLSC4": "UTILITIES", "CGAS5": "UTILITIES", "EEEL3": "UTILITIES",
    "CASN3": "UTILITIES", "ORVR3": "UTILITIES", "AMBP3": "UTILITIES",

    # --- CONSUMO NÃO CÍCLICO (Alimentos e Saúde) (25) ---
    "ABEV3": "CONSUMER_STAPLES", "JBSS3": "CONSUMER_STAPLES", "BRFS3": "CONSUMER_STAPLES", 
    "MRFG3": "CONSUMER_STAPLES", "BEEF3": "CONSUMER_STAPLES", "MDIA3": "CONSUMER_STAPLES", 
    "SMTO3": "CONSUMER_STAPLES", "CAML3": "CONSUMER_STAPLES", "SLCE3": "CONSUMER_STAPLES", 
    "ASAI3": "CONSUMER_STAPLES", "CRFB3": "CONSUMER_STAPLES", "GMAT3": "CONSUMER_STAPLES", 
    "NTCO3": "CONSUMER_STAPLES", "SOJA3": "CONSUMER_STAPLES", "AGXY3": "CONSUMER_STAPLES",
    "RDOR3": "HEALTHCARE", "RADL3": "HEALTHCARE", "HYPE3": "HEALTHCARE", "FLRY3": "HEALTHCARE", 
    "QUAL3": "HEALTHCARE", "BLAU3": "HEALTHCARE", "PARD3": "HEALTHCARE", "MATD3": "HEALTHCARE", 
    "ONCO3": "HEALTHCARE", "VVEO3": "HEALTHCARE",

    # --- CONSUMO CÍCLICO, VAREJO E CONSTRUÇÃO (40) ---
    "LREN3": "CONSUMER_DISCRETIONARY", "MGLU3": "CONSUMER_DISCRETIONARY", "AMER3": "CONSUMER_DISCRETIONARY", 
    "AZZA3": "CONSUMER_DISCRETIONARY", "SOMA3": "CONSUMER_DISCRETIONARY", "VIVA3": "CONSUMER_DISCRETIONARY", 
    "CEAB3": "CONSUMER_DISCRETIONARY", "LJQQ3": "CONSUMER_DISCRETIONARY", "COGN3": "CONSUMER_DISCRETIONARY", 
    "YDUQ3": "CONSUMER_DISCRETIONARY", "RENT3": "CONSUMER_DISCRETIONARY", "MOVI3": "CONSUMER_DISCRETIONARY", 
    "CYRE3": "CONSUMER_DISCRETIONARY", "EZTC3": "CONSUMER_DISCRETIONARY", "MRVE3": "CONSUMER_DISCRETIONARY", 
    "DIRR3": "CONSUMER_DISCRETIONARY", "CURY3": "CONSUMER_DISCRETIONARY", "PLPL3": "CONSUMER_DISCRETIONARY", 
    "EVEN3": "CONSUMER_DISCRETIONARY", "CSED3": "CONSUMER_DISCRETIONARY", "ANIM3": "CONSUMER_DISCRETIONARY", 
    "SEER3": "CONSUMER_DISCRETIONARY", "GRND3": "CONSUMER_DISCRETIONARY", "ARZZ3": "CONSUMER_DISCRETIONARY",
    "VAMO3": "CONSUMER_DISCRETIONARY", "SBFG3": "CONSUMER_DISCRETIONARY", "ALPA4": "CONSUMER_DISCRETIONARY",
    "MYPK3": "CONSUMER_DISCRETIONARY", "LEVE3": "CONSUMER_DISCRETIONARY", "TEND3": "CONSUMER_DISCRETIONARY",
    "JHSF3": "CONSUMER_DISCRETIONARY", "PDGR3": "CONSUMER_DISCRETIONARY", "MILS3": "CONSUMER_DISCRETIONARY",
    "UCAS3": "CONSUMER_DISCRETIONARY", "ESPA3": "CONSUMER_DISCRETIONARY", "MEAL3": "CONSUMER_DISCRETIONARY",
    "ZAMP3": "CONSUMER_DISCRETIONARY", "BKBR3": "CONSUMER_DISCRETIONARY", "AALR3": "CONSUMER_DISCRETIONARY",
    "GFSA3": "CONSUMER_DISCRETIONARY",

    # --- BENS INDUSTRIAIS E TRANSPORTE (20) ---
    "WEGE3": "INDUSTRIALS", "TASA4": "INDUSTRIALS", "RAIL3": "INDUSTRIALS", "CCRO3": "INDUSTRIALS", 
    "ECOR3": "INDUSTRIALS", "GOLL4": "INDUSTRIALS", "AZUL4": "INDUSTRIALS", "POMO4": "INDUSTRIALS", 
    "ROMI3": "INDUSTRIALS", "SHUL4": "INDUSTRIALS", "AERI3": "INDUSTRIALS", "VLID3": "INDUSTRIALS", 
    "SIMH3": "INDUSTRIALS", "GGPS3": "INDUSTRIALS", "PORT3": "INDUSTRIALS", "TGMA3": "INDUSTRIALS", 
    "LOGN3": "INDUSTRIALS", "FRAS3": "INDUSTRIALS", "RAPT4": "INDUSTRIALS", "TUPY3": "INDUSTRIALS",

    # --- TECNOLOGIA, COMUNICAÇÃO E REAL ESTATE (25) ---
    "TOTS3": "TECHNOLOGY", "LWSA3": "TECHNOLOGY", "CASH3": "TECHNOLOGY", "POSI3": "TECHNOLOGY", 
    "INTB3": "TECHNOLOGY", "IFCM3": "TECHNOLOGY", "VIVT3": "COMMUNICATIONS", "TIMS3": "COMMUNICATIONS", 
    "TELB4": "COMMUNICATIONS", "FIQE3": "TECHNOLOGY", "BRIT3": "TECHNOLOGY", "SQIA3": "TECHNOLOGY",
    "MULT3": "REAL_ESTATE", "ALSO3": "REAL_ESTATE", "LOGG3": "REAL_ESTATE", "LAVV3": "REAL_ESTATE", 
    "HBOR3": "REAL_ESTATE", "MDNE3": "REAL_ESTATE", "SYNE3": "REAL_ESTATE", "TRIS3": "REAL_ESTATE",
    "TECN3": "REAL_ESTATE", "MELK3": "REAL_ESTATE", "RNI3": "REAL_ESTATE", "MTRE3": "REAL_ESTATE",
    "BRPR3": "REAL_ESTATE"
}

# Lista de símbolos proxy (usada em alguns módulos antigos - pode manter)
PROXY_SYMBOLS = [
    "VALE3", "PETR4", "ITUB4", "BBDC4", "BBAS3",
    "ABEV3", "WEGE3", "JBSS3", "RENT3", "PRIO3",
    "SUZB3", "ELET3", "VIVT3", "HAPV3"
]

SCORE_WEIGHTS = {
    "EMA": 1.0,
    "RSI_ADX": 1.0,
    "VWAP": 1.0,
    "MACRO": 1.0,
    "ATR": 1.0,
    "CORR": 1.0
}

MIN_SIGNAL_SCORE = 40

# ===========================
# HORÁRIOS DE OPERAÇÃO
# ===========================
TRADING_START = "10:45"        # Após estabilização da abertura
TRADING_END   = "16:40"
FRIDAY_REDUCED_RISK_AFTER = "15:00"
NO_ENTRY_AFTER = "15:30"       # Fim das entradas (antes do fechamento nervoso)
CLOSE_ALL_BY = "16:50"         # FECHAMENTO FORÇADO (nunca posar no after)
DAILY_RESET_TIME = "10:15"     # Reset diário do circuit breaker

# ===========================
# GESTÃO DE RISCO
# ===========================
RISK_PER_TRADE_PCT = 0.01          # 1% da equity por trade padrão
REDUCED_RISK_PCT = 0.005           # 0.5% na sexta após 15h
MAX_DAILY_DRAWDOWN_PCT = 0.02       # Circuit breaker diário (2%)

MAX_RISK_PER_SYMBOL_PCT = 0.04     # Máximo 4% da equity por papel
MAX_SECTOR_EXPOSURE = 0.30        # Máx 30% do capital em 1 setor
MAX_SECTOR_EXPOSURE_PCT = 0.30     # Máx 30% do capital em 1 setor
SYMBOL_BLOCK_LOSS_PCT = 0.025      # Bloqueia ativo após perda de 2.5%
SYMBOL_BLOCK_HOURS = 72
SYMBOL_MAX_CONSECUTIVE_LOSSES = 3        # Bloqueia ativo após 3 perdas consecutivas
SYMBOL_COOLDOWN_HOURS = 24
# Slippage realista B3 (por liquidez/spread)
SLIPPAGE_MAP = {
    # Alta liquidez (top 10 volume B3)
    "PETR4": 0.0005, "VALE3": 0.0005, "ITUB4": 0.0006,
    "BBDC4": 0.0006, "BBAS3": 0.0007, "ABEV3": 0.0008,
    
    # Média liquidez (80% do SECTOR_MAP)
    "DEFAULT": 0.0015,  # 0.15% - realista para IOC em M15
    
}

MAX_TRADE_DURATION_CANDLES = 48    # Time-stop

# ===========================
# PYRAMIDING
# ===========================
ENABLE_PYRAMID = True
PYRAMID_MAX_LEGS = 2
PYRAMID_ATR_DISTANCE = 1.0         # Segunda perna só após +1.0 ATR a favor
PYRAMID_RISK_SPLIT = [0.6, 0.4]
PYRAMID_REQUIREMENTS = {
    "min_adx": 30,           # ADX > 30 (tendência forte confirmada)
    "max_rsi_long": 65,      # RSI não sobrecomprado (compra)
    "min_rsi_short": 35,     # RSI não sobrevendido (venda)
    "volume_ratio": 1.2,     # Volume 20% acima da média
    "time_since_entry": 30,  # Mínimo 30 min desde primeira perna
}

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
# GESTÃO AVANÇADA DE SAÍDA
# ===========================
ENABLE_BREAKEVEN = True
BREAKEVEN_ATR_MULT = 1.5          # Move SL para entrada após +1.5 ATR

ENABLE_PARTIAL_CLOSE = True
PARTIAL_CLOSE_ATR_MULT = 2.0      # Fecha 50% da posição em +2.0 ATR
PARTIAL_PERCENT = 0.5             # % da posição a fechar

ENABLE_TRAILING_STOP = True
TRAILING_ATR_MULT_INITIAL = 2.5   # Trailing inicial
TRAILING_ATR_MULT_TIGHT = 1.5     # Aperta após +3 ATR
# ===========================
# NOTIFICAÇÕES TELEGRAM
# ===========================
ENABLE_TELEGRAM_NOTIF = True
TELEGRAM_BOT_TOKEN = "8551934559:AAGZRMxH51N-IcsAuFJzelafOuVo1pMS9nI"          # Ex: 123456789:AAF...
TELEGRAM_CHAT_ID = 8400631213  
EOD_REPORT_ENABLED = True
EOD_REPORT_TIME = "17:55"                 # Seu chat_id (número inteiro)
# ===========================

# =========================
# ⏰ TIME-AWARE SCORING
# =========================

TIME_SCORE_RULES = {
    "OPEN": {
        "start": "10:00",
        "end":   "11:30",
        "adx_min": 15,
        "min_score": 40,
        "atr_max": 8.0,
    },
    "MID": {
        "start": "11:30",
        "end":   "14:30",
        "adx_min": 12,
        "min_score": 30,
        "atr_max": 10.0,
    },
    "LATE": {
        "start": "14:30",
        "end":   "16:55",
        "adx_min": 10,
        "min_score": 20,
        "atr_max": 15.0,
    }
}

# =========================
# ⚡ POWER-HOUR MODE
# =========================

POWER_HOUR = {
    "enabled": True,
    "start": "15:30",
    "end":   "16:55",
    "min_atr_pct": 0.8,
    "min_volume_ratio": 1.1,
    "score_boost": 15,
}

# =========================
# 🚀 VOLATILITY BREAKOUT
# =========================

VOL_BREAKOUT = {
    "enabled": True,
    "lookback": 20,
    "atr_expansion": 1.15,
    "volume_ratio": 1.2,
    "score_boost": 20,
}


# ===========================
# PARÂMETROS OTIMIZADOS MANUAIS (ELITE)
# ===========================
ELITE_SYMBOLS = {
    "IRBR3": {'ema_short': 25, 'ema_long': 40, 'rsi_low': 45, 'rsi_high': 80, 'adx_threshold': 37, 'mom_min': 0.07},
    "BEES3": {'ema_short': 32, 'ema_long': 118, 'rsi_low': 40, 'rsi_high': 79, 'adx_threshold': 23, 'mom_min': 0.1},
    "PETR4": {'ema_short': 30, 'ema_long': 62, 'rsi_low': 20, 'rsi_high': 74, 'adx_threshold': 37, 'mom_min': 0.15},
    "PETR3": {'ema_short': 34, 'ema_long': 75, 'rsi_low': 24, 'rsi_high': 55, 'adx_threshold': 32, 'mom_min': 0.15},
    "PRIO3": {'ema_short': 21, 'ema_long': 49, 'rsi_low': 24, 'rsi_high': 65, 'adx_threshold': 16, 'mom_min': 0.1},
    "OPCT3": {'ema_short': 21, 'ema_long': 70, 'rsi_low': 30, 'rsi_high': 60, 'adx_threshold': 21, 'mom_min': 0.020000000000000018},
    "VALE3": {'ema_short': 5, 'ema_long': 49, 'rsi_low': 29, 'rsi_high': 58, 'adx_threshold': 33, 'mom_min': 0.1},
    "SUZB3": {'ema_short': 34, 'ema_long': 101, 'rsi_low': 42, 'rsi_high': 71, 'adx_threshold': 24, 'mom_min': 0.0},
    "CSNA3": {'ema_short': 35, 'ema_long': 70, 'rsi_low': 32, 'rsi_high': 56, 'adx_threshold': 26, 'mom_min': 0.07},
    "USIM5": {'ema_short': 31, 'ema_long': 78, 'rsi_low': 43, 'rsi_high': 61, 'adx_threshold': 29, 'mom_min': 0.03},
    "BRKM5": {'ema_short': 35, 'ema_long': 110, 'rsi_low': 26, 'rsi_high': 76, 'adx_threshold': 33, 'mom_min': 0.05000000000000002},
    "CMIG4": {'ema_short': 25, 'ema_long': 49, 'rsi_low': 25, 'rsi_high': 79, 'adx_threshold': 39, 'mom_min': -0.039999999999999994},
    "SAPR11": {'ema_short': 31, 'ema_long': 68, 'rsi_low': 45, 'rsi_high': 55, 'adx_threshold': 38, 'mom_min': -0.12},
    "LIGT3": {'ema_short': 14, 'ema_long': 70, 'rsi_low': 31, 'rsi_high': 74, 'adx_threshold': 40, 'mom_min': -0.009999999999999981},
    "CMIG3": {'ema_short': 32, 'ema_long': 93, 'rsi_low': 33, 'rsi_high': 80, 'adx_threshold': 37, 'mom_min': 0.08000000000000002},
    "ASAI3": {'ema_short': 27, 'ema_long': 76, 'rsi_low': 42, 'rsi_high': 66, 'adx_threshold': 38, 'mom_min': -0.039999999999999994},
    "HYPE3": {'ema_short': 8, 'ema_long': 79, 'rsi_low': 35, 'rsi_high': 65, 'adx_threshold': 31, 'mom_min': 0.09},
    "VVEO3": {'ema_short': 13, 'ema_long': 118, 'rsi_low': 41, 'rsi_high': 55, 'adx_threshold': 19, 'mom_min': -0.12},
    "VIVA3": {'ema_short': 35, 'ema_long': 120, 'rsi_low': 35, 'rsi_high': 72, 'adx_threshold': 24, 'mom_min': 0.15},
    "SBFG3": {'ema_short': 31, 'ema_long': 90, 'rsi_low': 28, 'rsi_high': 65, 'adx_threshold': 18, 'mom_min': -0.10999999999999999},
    "VAMO3": {'ema_short': 35, 'ema_long': 113, 'rsi_low': 27, 'rsi_high': 80, 'adx_threshold': 34, 'mom_min': -0.13},
    "MEAL3": {'ema_short': 10, 'ema_long': 47, 'rsi_low': 31, 'rsi_high': 74, 'adx_threshold': 18, 'mom_min': -0.09999999999999999},
    "AALR3": {'ema_short': 30, 'ema_long': 91, 'rsi_low': 32, 'rsi_high': 64, 'adx_threshold': 36, 'mom_min': 0.1},
    "LOGN3": {'ema_short': 35, 'ema_long': 97, 'rsi_low': 29, 'rsi_high': 58, 'adx_threshold': 35, 'mom_min': -0.039999999999999994},
    "CASH3": {'ema_short': 13, 'ema_long': 97, 'rsi_low': 24, 'rsi_high': 57, 'adx_threshold': 40, 'mom_min': -0.06},
    "TIMS3": {'ema_short': 18, 'ema_long': 83, 'rsi_low': 42, 'rsi_high': 73, 'adx_threshold': 15, 'mom_min': 0.12000000000000002},
    "TRIS3": {'ema_short': 22, 'ema_long': 116, 'rsi_low': 24, 'rsi_high': 74, 'adx_threshold': 23, 'mom_min': 0.13999999999999999},
}

LOW_LIQUIDITY_SYMBOLS = {

}