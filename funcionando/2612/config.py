import os

# ===========================
# PARÂMETROS DO PROJETO XP3/B3
# ===========================
MT5_TERMINAL_PATH = r"C:\MetaTrader 5 Terminal\terminal64.exe"
# Limites gerais do bot
MAX_SYMBOLS = 10
MAX_PER_SECTOR = 2
MAX_CORRELATION_PAIRS = 3

# Mapa de setores (todos os ativos monitorados)
SECTOR_MAP = {
    # FINANCEIRO (25)
    "ITUB4": "FINANCEIRO",
    "BBDC4": "FINANCEIRO",
    "BBAS3": "FINANCEIRO",
    "SANB11": "FINANCEIRO",
    "B3SA3": "FINANCEIRO",
    "BPAC11": "FINANCEIRO",
    "ABCB4": "FINANCEIRO",
    "PSSA3": "FINANCEIRO",
    "ITSA4": "FINANCEIRO",
    "BBSE3": "FINANCEIRO",
    "CXSE3": "FINANCEIRO",
    "WIZC3": "FINANCEIRO",
    "IRBR3": "FINANCEIRO",
    "TRAD3": "FINANCEIRO",
    "CASH3": "FINANCEIRO",
    "MODL11": "FINANCEIRO",
    "BRSR6": "FINANCEIRO",
    "PINE4": "FINANCEIRO",
    "IGTI11": "FINANCEIRO",
    "BPAN4": "FINANCEIRO",
    "BMGB4": "FINANCEIRO",
    "CIEL3": "FINANCEIRO",
    "SANB4": "FINANCEIRO",
    "ITUB3": "FINANCEIRO",
    "BBDC3": "FINANCEIRO",
    # ENERGIA / PETRÓLEO (20)
    "PETR4": "ENERGIA",
    "PETR3": "ENERGIA",
    "PRIO3": "ENERGIA",
    "CSAN3": "ENERGIA",
    "VBBR3": "ENERGIA",
    "UGPA3": "ENERGIA",
    "EQTL3": "ENERGIA",
    "ELET3": "ENERGIA",
    "ELET6": "ENERGIA",
    "ENEV3": "ENERGIA",
    "CPLE6": "ENERGIA",
    "ENGI11": "ENERGIA",
    "TAEE11": "ENERGIA",
    "CMIG4": "ENERGIA",
    "SBSP3": "ENERGIA",
    "CSMG3": "ENERGIA",
    "NEOE3": "ENERGIA",
    "LIGT3": "ENERGIA",
    "SAPR11": "ENERGIA",
    "EGIE3": "ENERGIA",
    # MATERIAIS BÁSICOS / MINERAÇÃO (18)
    "VALE3": "MATERIAIS BÁSICOS",
    "GGBR4": "MATERIAIS BÁSICOS",
    "USIM5": "MATERIAIS BÁSICOS",
    "CSNA3": "MATERIAIS BÁSICOS",
    "SUZB3": "MATERIAIS BÁSICOS",
    "KLBN11": "MATERIAIS BÁSICOS",
    "BRKM5": "MATERIAIS BÁSICOS",
    "CMIN3": "MATERIAIS BÁSICOS",
    "GOAU4": "MATERIAIS BÁSICOS",
    "AURA33": "MATERIAIS BÁSICOS",
    "GGBR3": "MATERIAIS BÁSICOS",
    "USIM3": "MATERIAIS BÁSICOS",
    "BRAP4": "MATERIAIS BÁSICOS",
    "FESA4": "MATERIAIS BÁSICOS",
    "UNIP6": "MATERIAIS BÁSICOS",
    "DXCO3": "MATERIAIS BÁSICOS",
    "RANI3": "MATERIAIS BÁSICOS",
    "EUCA4": "MATERIAIS BÁSICOS",
    # CONSUMO NÃO CÍCLICO (18)
    "ABEV3": "CONSUMO NÃO CÍCLICO",
    "JBSS3": "CONSUMO NÃO CÍCLICO",
    "ASAI3": "CONSUMO NÃO CÍCLICO",
    "CRFB3": "CONSUMO NÃO CÍCLICO",
    "RADL3": "CONSUMO NÃO CÍCLICO",
    "PCAR3": "CONSUMO NÃO CÍCLICO",
    "MRFG3": "CONSUMO NÃO CÍCLICO",
    "BRFS3": "CONSUMO NÃO CÍCLICO",
    "NTCO3": "CONSUMO NÃO CÍCLICO",
    "SMTO3": "CONSUMO NÃO CÍCLICO",
    "BEEF3": "CONSUMO NÃO CÍCLICO",
    "SLCE3": "CONSUMO NÃO CÍCLICO",
    "MDIA3": "CONSUMO NÃO CÍCLICO",
    "GMAT3": "CONSUMO NÃO CÍCLICO",
    "HAPV3": "CONSUMO NÃO CÍCLICO",
    "RAIL3": "CONSUMO NÃO CÍCLICO",
    "RAIZ4": "CONSUMO NÃO CÍCLICO",
    "JALL3": "CONSUMO NÃO CÍCLICO",
    # SAÚDE (12)
    "HAPV3": "SAÚDE",
    "RDOR3": "SAÚDE",
    "FLRY3": "SAÚDE",
    "QUAL3": "SAÚDE",
    "BLAU3": "SAÚDE",
    "HYPE3": "SAÚDE",
    "PARD3": "SAÚDE",
    "ONCO3": "SAÚDE",
    "VVEO3": "SAÚDE",
    "KRSA3": "SAÚDE",
    "MATD3": "SAÚDE",
    "DASA3": "SAÚDE",
    # CONSUMO CÍCLICO / VAREJO (25)
    "LREN3": "CONSUMO CÍCLICO",
    "MGLU3": "CONSUMO CÍCLICO",
    "AMER3": "CONSUMO CÍCLICO",
    "SOMA3": "CONSUMO CÍCLICO",
    "ARZZ3": "CONSUMO CÍCLICO",
    "VIVA3": "CONSUMO CÍCLICO",
    "CEAB3": "CONSUMO CÍCLICO",
    "GUAR3": "CONSUMO CÍCLICO",
    "CYRE3": "CONSUMO CÍCLICO",
    "MRVE3": "CONSUMO CÍCLICO",
    "CURY3": "CONSUMO CÍCLICO",
    "TEND3": "CONSUMO CÍCLICO",
    "YDUQ3": "CONSUMO CÍCLICO",
    "COGN3": "CONSUMO CÍCLICO",
    "ZAMP3": "CONSUMO CÍCLICO",
    "VAMO3": "CONSUMO CÍCLICO",
    "AZUL4": "CONSUMO CÍCLICO",
    "GOLL4": "CONSUMO CÍCLICO",
    "JHSF3": "CONSUMO CÍCLICO",
    "EVEN3": "CONSUMO CÍCLICO",
    "DIRR3": "CONSUMO CÍCLICO",
    "ANIM3": "CONSUMO CÍCLICO",
    "SBFG3": "CONSUMO CÍCLICO",
    "ALSO3": "CONSUMO CÍCLICO",
    "MEAL3": "CONSUMO CÍCLICO",
    # INDUSTRIAL / TRANSPORTE (15)
    "WEGE3": "INDUSTRIAL",
    "EMBR3": "INDUSTRIAL",
    "CCRO3": "INDUSTRIAL",
    "RAIL3": "INDUSTRIAL",
    "ECOR3": "INDUSTRIAL",
    "TUPY3": "INDUSTRIAL",
    "ROMI3": "INDUSTRIAL",
    "RENT3": "INDUSTRIAL",
    "AZUL4": "INDUSTRIAL",
    "GOLL4": "INDUSTRIAL",
    "LOGN3": "INDUSTRIAL",
    "VAMO3": "INDUSTRIAL",
    "KEPL3": "INDUSTRIAL",
    "FRAS3": "INDUSTRIAL",
    "AERI3": "INDUSTRIAL",
    # TECNOLOGIA / COMUNICAÇÃO (12)
    "TOTS3": "TECNOLOGIA",
    "LWSA3": "TECNOLOGIA",
    "POSI3": "TECNOLOGIA",
    "INTB3": "TECNOLOGIA",
    "SQIA3": "TECNOLOGIA",
    "NGRD3": "TECNOLOGIA",
    "VIVT3": "COMUNICAÇÕES",
    "TIMS3": "COMUNICAÇÕES",
    "DESK3": "TECNOLOGIA",
    "CASH3": "TECNOLOGIA",
    "BMKS3": "TECNOLOGIA",
    "IFCM3": "TECNOLOGIA",
    # OUTROS / DIVERSIFICADOS (5)
    "BRML3": "IMOBILIÁRIO",
    "MULT3": "IMOBILIÁRIO",
    "JSRE11": "IMOBILIÁRIO",
    "HGLG11": "IMOBILIÁRIO",
    "VISC11": "IMOBILIÁRIO",
}
# Lista de símbolos proxy (usada em alguns módulos antigos - pode manter)
PROXY_SYMBOLS = [
    "VALE3",
    "PETR4",
    "ITUB4",
    "BBDC4",
    "BBAS3",
    "ABEV3",
    "WEGE3",
    "JBSS3",
    "RENT3",
    "PRIO3",
    "SUZB3",
    "ELET3",
    "VIVT3",
    "HAPV3",
]

SCORE_WEIGHTS = {
    "EMA": 1.0,
    "RSI_ADX": 1.0,
    "VWAP": 1.0,
    "MACRO": 1.0,
    "ATR": 1.0,
    "CORR": 1.0,
}

MIN_SIGNAL_SCORE = 35

# ===========================
# HORÁRIOS DE OPERAÇÃO
# ===========================
TRADING_START = "10:30"  # Após estabilização da abertura
TRADING_END = "16:40"
FRIDAY_REDUCED_RISK_AFTER = "15:30"
NO_ENTRY_AFTER = "16:00"  # Fim das entradas (antes do fechamento nervoso)
CLOSE_ALL_BY = "16:45"  # FECHAMENTO FORÇADO (nunca posar no after)
DAILY_RESET_TIME = "10:15"  # Reset diário do circuit breaker

# ===========================
# GESTÃO DE RISCO
# ===========================
RISK_PER_TRADE_PCT = 0.006  # 1% da equity por trade padrão
REDUCED_RISK_PCT = 0.005  # 0.5% na sexta após 15h
MAX_DAILY_DRAWDOWN_PCT = 0.015  # Circuit breaker diário (2%)

MAX_RISK_PER_SYMBOL_PCT = 0.04  # Máximo 4% da equity por papel
MAX_SECTOR_EXPOSURE = 0.30  # Máx 30% do capital em 1 setor
MAX_SECTOR_EXPOSURE_PCT = 0.25  # Máx 30% do capital em 1 setor
SYMBOL_BLOCK_LOSS_PCT = 0.025  # Bloqueia ativo após perda de 2.5%
SYMBOL_BLOCK_HOURS = 72
SYMBOL_MAX_CONSECUTIVE_LOSSES = 2  # Bloqueia ativo após 3 perdas consecutivas
SYMBOL_COOLDOWN_HOURS = 24
# Slippage realista B3 (por liquidez/spread)
SLIPPAGE_MAP = {
    # Alta liquidez (top 10 volume B3)
    "PETR4": 0.0005,
    "VALE3": 0.0005,
    "ITUB4": 0.0006,
    "BBDC4": 0.0006,
    "BBAS3": 0.0007,
    "ABEV3": 0.0008,
    # Média liquidez (80% do SECTOR_MAP)
    "DEFAULT": 0.0015,  # 0.15% - realista para IOC em M15
}

MAX_TRADE_DURATION_CANDLES = 40  # Time-stop

# ===========================
# PYRAMIDING
# ===========================
ENABLE_PYRAMID = True
PYRAMID_MAX_LEGS = 2
PYRAMID_ATR_DISTANCE = 1.0  # Segunda perna só após +1.0 ATR a favor
PYRAMID_RISK_SPLIT = [0.6, 0.4]
PYRAMID_REQUIREMENTS = {
    "min_adx": 30,  # ADX > 30 (tendência forte confirmada)
    "max_rsi_long": 65,  # RSI não sobrecomprado (compra)
    "min_rsi_short": 35,  # RSI não sobrevendido (venda)
    "volume_ratio": 1.2,  # Volume 20% acima da média
    "time_since_entry": 30,  # Mínimo 30 min desde primeira perna
}

# ===========================
# STOP LOSS / TAKE PROFIT
# ===========================
SL_ATR_MULTIPLIER = 2.0  # SL inicial = preço ± ATR × 2.0
TP_ATR_MULT = 3.0  # TP opcional (não usado atualmente, mas disponível)
TRAILING_STEP_ATR_MULTIPLIER = 1.0

# ===========================
# FILTROS PROFISSIONAIS B3
# ===========================
MIN_AVG_VOLUME_20 = 300000  # Volume médio 20 períodos mínimo
MAX_GAP_OPEN_PCT = 0.03  # Gap de abertura > 3% → bloqueia entrada

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
BREAKEVEN_ATR_MULT = 2.0  # Move SL para entrada após +1.5 ATR

ENABLE_PARTIAL_CLOSE = True
PARTIAL_CLOSE_ATR_MULT = 3.0  # Fecha 50% da posição em +2.0 ATR
PARTIAL_PERCENT = 0.5  # % da posição a fechar
MAX_TRADE_DURATION_CANDLES = 80
ENABLE_TRAILING_STOP = True
TRAILING_ATR_MULT_INITIAL = 2.5  # Trailing inicial
TRAILING_ATR_MULT_TIGHT = 1.3 # Aperta após +3 ATR
# ===========================
# NOTIFICAÇÕES TELEGRAM
# ===========================
ENABLE_TELEGRAM_NOTIF = True
TELEGRAM_BOT_TOKEN = (
    "8551934559:AAGZRMxH51N-IcsAuFJzelafOuVo1pMS9nI"  # Ex: 123456789:AAF...
)
TELEGRAM_CHAT_ID = 8400631213
EOD_REPORT_ENABLED = True
EOD_REPORT_TIME = "17:55"  # Seu chat_id (número inteiro)
# ===========================

# =========================
# ⏰ TIME-AWARE SCORING
# =========================

TIME_SCORE_RULES = {
    "OPEN": {
        "start": "10:00",
        "end": "11:30",
        "adx_min": 10,
        "min_score": 30,
        "atr_max": 10.0,
    },
    "MID": {
        "start": "11:30",
        "end": "14:30",
        "adx_min": 8,
        "min_score": 25,
        "atr_max": 12.0,
    },
    "LATE": {
        "start": "14:30",
        "end": "16:55",
        "adx_min": 6,
        "min_score": 20,
        "atr_max": 15.0,
    },
}

ADAPTIVE_FILTERS = {
    "spread": {
        "normal": 0.15,  # 10:00-15:30 (era 0.10)
        "power_hour": 0.30,  # 15:30-18:00 (era 0.12)
    },
    "book_depth": {
        "normal": 0.50,  # Exige 50% do volume
        "power_hour": 0.20,  # Exige apenas 20%
    },
    "volume_impact": {
        "normal": 0.20,  # Máx 20% do volume médio
        "power_hour": 0.35,  # Máx 35% (maior tolerância)
    },
}

# =========================
# ⚡ POWER-HOUR MODE
# =========================

POWER_HOUR = {
    "enabled": True,
    "start": "15:30",
    "end": "16:55",
    "min_atr_pct": 0.8,
    "min_volume_ratio": 1.1,
    "score_boost": 10,
}

# =========================
# 🚀 VOLATILITY BREAKOUT
# =========================

VOL_BREAKOUT = {
    "enabled": True,
    "lookback": 20,
    "atr_expansion": 1.25,
    "volume_ratio": 1.2,
    "score_boost": 15,
}

# ===========================
# 🎯 TARGETS DINÂMICOS POR REGIME
# ===========================

TP_RULES = {
    "TRENDING": {
        "min_adx": 30,
        "tp_mult": 4.5,  # 1:2.25 R:R
        "partial_mult": 2.5,
        "trailing_initial": 3.0,
        "trailing_tight": 1.8,
    },
    "RANGING": {
        "min_adx": 0,
        "max_adx": 25,
        "tp_mult": 2.0,  # 1:1.25 R:R (conservador)
        "partial_mult": 1.8,
        "trailing_initial": 2.0,
        "trailing_tight": 1.2,
    },
    "BREAKOUT": {
        "vol_expansion": 1.3,  # ATR 30% acima da média
        "tp_mult": 5.0,  # 1:2.5 R:R (agressivo)
        "partial_mult": 3.0,
        "trailing_initial": 3.5,
        "trailing_tight": 2.0,
    },
}

# ===========================
# 📈 METAS DE PROFIT FACTOR
# ===========================

PROFIT_TARGETS = {
    "daily": {
        "min_pf": 1.5,  # Profit Factor mínimo do dia
        "target_return": 0.015,  # 1.5% ao dia
        "max_return": 0.03,  # 3% ao dia (conservadorismo)
    },
    "weekly": {
        "target_return": 0.06,  # 6% na semana
        "max_dd": 0.04,  # 4% de drawdown máximo
    },
    "monthly": {
        "target_return": 0.10,  # 20% ao mês (agressivo mas possível)
        "min_sharpe": 1.5,  # Sharpe Ratio > 1.5
    },
}

# ===========================
# 🛡️ PROFIT PROTECTION
# ===========================

PROFIT_LOCK = {
    "enabled": True,
    "daily_target_pct": 0.01,  # 2% de lucro no dia
    "lock_pct": 0.70,  # Trava 70% do lucro
    "reduce_risk": True,  # Reduz risco para 0.5% após meta
}

# ===========================
# PARÂMETROS OTIMIZADOS MANUAIS (ELITE)
# ===========================
ELITE_SYMBOLS = {
    "USIM5": {
        "ema_short": 23,
        "ema_long": 199,
        "rsi_low": 50,
        "rsi_high": 77,
        "adx_threshold": 28,
        "mom_min": 5.138948139241887e-06,
    },
    "IRBR3": {
        "ema_short": 8,
        "ema_long": 163,
        "rsi_low": 46,
        "rsi_high": 80,
        "adx_threshold": 24,
        "mom_min": 0.0018846003768108717,
    },
    "AURA33": {
        "ema_short": 17,
        "ema_long": 181,
        "rsi_low": 43,
        "rsi_high": 82,
        "adx_threshold": 20,
        "mom_min": 0.00048275986007657656,
    },
    "PCAR3": {
        "ema_short": 11,
        "ema_long": 41,
        "rsi_low": 44,
        "rsi_high": 65,
        "adx_threshold": 21,
        "mom_min": 0.0013395658640018304,
    },
    "TUPY3": {
        "ema_short": 18,
        "ema_long": 186,
        "rsi_low": 54,
        "rsi_high": 75,
        "adx_threshold": 20,
        "mom_min": 0.0013592259460763374,
    },
    "LREN3": {
        "ema_short": 12,
        "ema_long": 177,
        "rsi_low": 46,
        "rsi_high": 81,
        "adx_threshold": 23,
        "mom_min": 0.0002765389725208764,
    },
    "JHSF3": {
        "ema_short": 23,
        "ema_long": 195,
        "rsi_low": 49,
        "rsi_high": 81,
        "adx_threshold": 25,
        "mom_min": 0.0009653254445272919,
    },
    "CSMG3": {
        "ema_short": 13,
        "ema_long": 114,
        "rsi_low": 55,
        "rsi_high": 78,
        "adx_threshold": 30,
        "mom_min": 0.0007195374265626512,
    },
    "SANB11": {
        "ema_short": 19,
        "ema_long": 49,
        "rsi_low": 55,
        "rsi_high": 80,
        "adx_threshold": 25,
        "mom_min": 0.0017552843261563168,
    },
    "CASH3": {
        "ema_short": 11,
        "ema_long": 62,
        "rsi_low": 47,
        "rsi_high": 65,
        "adx_threshold": 25,
        "mom_min": 0.0014979155304656155,
    },
    "CYRE3": {
        "ema_short": 12,
        "ema_long": 185,
        "rsi_low": 53,
        "rsi_high": 85,
        "adx_threshold": 24,
        "mom_min": 0.0011077793168624331,
    },
    "BBAS3": {
        "ema_short": 15,
        "ema_long": 150,
        "rsi_low": 53,
        "rsi_high": 85,
        "adx_threshold": 25,
        "mom_min": 0.0012276588504268758,
    },
    "CMIG4": {
        "ema_short": 17,
        "ema_long": 96,
        "rsi_low": 48,
        "rsi_high": 81,
        "adx_threshold": 28,
        "mom_min": 0.0014777043061999216,
    },
    "CSNA3": {
        "ema_short": 15,
        "ema_long": 44,
        "rsi_low": 48,
        "rsi_high": 65,
        "adx_threshold": 27,
        "mom_min": 0.0017812093811960022,
    },
}

LOW_LIQUIDITY_SYMBOLS = {}
