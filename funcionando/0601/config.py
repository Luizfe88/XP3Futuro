import os

# ===========================
# PARÂMETROS DO PROJETO XP3/B3
# ===========================
MT5_TERMINAL_PATH = r"C:\MetaTrader 5 Terminal\terminal64.exe"
# Limites gerais do bot
MAX_SYMBOLS = 8
MAX_PER_SECTOR = 2
MAX_CORRELATION_PAIRS = 2
# 🆕 LIMITES POR SUBSETOR (dentro de FINANCEIRO)
MAX_PER_SUBSETOR = {
    "BANCOS": 2,  # Máx 2 bancos (ITUB4, BBDC4, BBAS3, etc)
    "CORRETORAS": 1,  # Máx 1 corretora (B3SA3, BPAC11)
    "SEGUROS": 1,  # Máx 1 seguro (IRBR3, PSSA3)
}

# 🆕 Mapa de subsetores
SUBSETOR_MAP = {
    # BANCOS (máx 2)
    "ITUB4": "BANCOS", "BBDC4": "BANCOS", "BBDC3": "BANCOS", 
    "BBAS3": "BANCOS", "SANB11": "BANCOS", "BPAN4": "BANCOS",
    
    # CORRETORAS (máx 1)
    "B3SA3": "CORRETORAS", "BPAC11": "CORRETORAS",
    
    # SEGUROS (máx 1)
    "IRBR3": "SEGUROS", "PSSA3": "SEGUROS",
}

# Mapa de setores (todos os ativos monitorados)
SECTOR_MAP = {
    # FINANCEIRO (32)
    "ITUB4": "FINANCEIRO", "ITUB3": "FINANCEIRO", "BBDC4": "FINANCEIRO", "BBDC3": "FINANCEIRO",
    "BBAS3": "FINANCEIRO", "SANB11": "FINANCEIRO", "SANB3": "FINANCEIRO", "SANB4": "FINANCEIRO",
    "B3SA3": "FINANCEIRO", "BPAC11": "FINANCEIRO", "ITSA4": "FINANCEIRO", "ITSA3": "FINANCEIRO",
    "BBSE3": "FINANCEIRO", "CXSE3": "FINANCEIRO", "PSSA3": "FINANCEIRO", "IRBR3": "FINANCEIRO",
    "ABCB4": "FINANCEIRO", "BEES3": "FINANCEIRO", "BEES4": "FINANCEIRO", "BRSR6": "FINANCEIRO",
    "PINE4": "FINANCEIRO", "BPAN4": "FINANCEIRO", "BMGB4": "FINANCEIRO", "BIDI11": "FINANCEIRO",
    "TRAD3": "FINANCEIRO", "WIZC3": "FINANCEIRO", "BBAV3": "FINANCEIRO", "BNBR3": "FINANCEIRO",
    "BRIV4": "FINANCEIRO", "RPAD5": "FINANCEIRO", "CRIV4": "FINANCEIRO", "BAZA3": "FINANCEIRO",

    # ENERGIA / UTILIDADES / SANEAMENTO (35)
    "PETR4": "ENERGIA", "PETR3": "ENERGIA", "PRIO3": "ENERGIA", "RECV3": "ENERGIA",
    "RRRP3": "ENERGIA", "CSAN3": "ENERGIA", "VBBR3": "ENERGIA", "UGPA3": "ENERGIA",
    "ELET3": "ENERGIA", "ELET6": "ENERGIA", "EQTL3": "ENERGIA", "CPLE6": "ENERGIA",
    "CPLE3": "ENERGIA", "CPFE3": "ENERGIA", "ENGI11": "ENERGIA", "TAEE11": "ENERGIA",
    "CMIG4": "ENERGIA", "CMIG3": "ENERGIA", "TRPL4": "ENERGIA", "EGIE3": "ENERGIA",
    "AURE3": "ENERGIA", "ENEV3": "ENERGIA", "NEOE3": "ENERGIA", "AESB3": "ENERGIA",
    "LIGT3": "ENERGIA", "SBSP3": "ENERGIA", "CSMG3": "ENERGIA", "SAPR11": "ENERGIA",
    "SAPR4": "ENERGIA", "ALUP11": "ENERGIA", "MEGA3": "ENERGIA", "KEPL3": "ENERGIA",
    "RUM3": "ENERGIA", "CEEB3": "ENERGIA", "EKTR4": "ENERGIA",

    # MATERIAIS BÁSICOS / MINERAÇÃO (22)
    "VALE3": "MATERIAIS BÁSICOS", "GGBR4": "MATERIAIS BÁSICOS", "GGBR3": "MATERIAIS BÁSICOS",
    "GOAU4": "MATERIAIS BÁSICOS", "USIM5": "MATERIAIS BÁSICOS", "USIM3": "MATERIAIS BÁSICOS",
    "CSNA3": "MATERIAIS BÁSICOS", "CMIN3": "MATERIAIS BÁSICOS", "BRAP4": "MATERIAIS BÁSICOS",
    "SUZB3": "MATERIAIS BÁSICOS", "KLBN11": "MATERIAIS BÁSICOS", "KLBN4": "MATERIAIS BÁSICOS",
    "BRKM5": "MATERIAIS BÁSICOS", "FESA4": "MATERIAIS BÁSICOS", "UNIP6": "MATERIAIS BÁSICOS",
    "DXCO3": "MATERIAIS BÁSICOS", "RANI3": "MATERIAIS BÁSICOS", "EUCA4": "MATERIAIS BÁSICOS",
    "AURA33": "MATERIAIS BÁSICOS", "CBAV3": "MATERIAIS BÁSICOS", "TASA4": "MATERIAIS BÁSICOS",
    "CRPG5": "MATERIAIS BÁSICOS",

    # CONSUMO NÃO CÍCLICO / AGRO (22)
    "ABEV3": "CONSUMO NÃO CÍCLICO", "JBSS3": "CONSUMO NÃO CÍCLICO", "BRFS3": "CONSUMO NÃO CÍCLICO",
    "MRFG3": "CONSUMO NÃO CÍCLICO", "BEEF3": "CONSUMO NÃO CÍCLICO", "ASAI3": "CONSUMO NÃO CÍCLICO",
    "CRFB3": "CONSUMO NÃO CÍCLICO", "PCAR3": "CONSUMO NÃO CÍCLICO", "GMAT3": "CONSUMO NÃO CÍCLICO",
    "NTCO3": "CONSUMO NÃO CÍCLICO", "SMTO3": "CONSUMO NÃO CÍCLICO", "SLCE3": "CONSUMO NÃO CÍCLICO",
    "RAIZ4": "CONSUMO NÃO CÍCLICO", "MDIA3": "CONSUMO NÃO CÍCLICO", "CAML3": "CONSUMO NÃO CÍCLICO",
    "SOJA3": "CONSUMO NÃO CÍCLICO", "AGRO3": "CONSUMO NÃO CÍCLICO", "JALL3": "CONSUMO NÃO CÍCLICO",
    "FRTA3": "CONSUMO NÃO CÍCLICO", "POMO4": "CONSUMO NÃO CÍCLICO", "MDNE3": "CONSUMO NÃO CÍCLICO",
    "ORVR3": "CONSUMO NÃO CÍCLICO",

    # SAÚDE (15)
    "RDOR3": "SAÚDE", "HAPV3": "SAÚDE", "RADL3": "SAÚDE", "FLRY3": "SAÚDE",
    "HYPE3": "SAÚDE", "ONCO3": "SAÚDE", "QUAL3": "SAÚDE", "BLAU3": "SAÚDE",
    "VVEO3": "SAÚDE", "MATD3": "SAÚDE", "DASA3": "SAÚDE", "ODPV3": "SAÚDE",
    "PARD3": "SAÚDE", "AALR3": "SAÚDE", "KRSA3": "SAÚDE",

    # CONSUMO CÍCLICO / VAREJO / EDUCAÇÃO (30)
    "LREN3": "CONSUMO CÍCLICO", "MGLU3": "CONSUMO CÍCLICO", "AMER3": "CONSUMO CÍCLICO",
    "ARZZ3": "CONSUMO CÍCLICO", "VIVA3": "CONSUMO CÍCLICO", "CEAB3": "CONSUMO CÍCLICO",
    "GUAR3": "CONSUMO CÍCLICO", "SBFG3": "CONSUMO CÍCLICO", "AMBP3": "CONSUMO CÍCLICO",
    "ALPA4": "CONSUMO CÍCLICO", "LJQQ3": "CONSUMO CÍCLICO", "VIIA3": "CONSUMO CÍCLICO",
    "YDUQ3": "CONSUMO CÍCLICO", "COGN3": "CONSUMO CÍCLICO", "ANIM3": "CONSUMO CÍCLICO",
    "SEER3": "CONSUMO CÍCLICO", "CYRE3": "CONSUMO CÍCLICO", "MRVE3": "CONSUMO CÍCLICO",
    "CURY3": "CONSUMO CÍCLICO", "TEND3": "CONSUMO CÍCLICO", "DIRR3": "CONSUMO CÍCLICO",
    "EVEN3": "CONSUMO CÍCLICO", "JHSF3": "CONSUMO CÍCLICO", "EZTC3": "CONSUMO CÍCLICO",
    "PLPL3": "CONSUMO CÍCLICO", "MTRE3": "CONSUMO CÍCLICO", "ZAMP3": "CONSUMO CÍCLICO",
    "MEAL3": "CONSUMO CÍCLICO", "BKBR3": "CONSUMO CÍCLICO", "GRND3": "CONSUMO CÍCLICO",

    # INDUSTRIAL / LOGÍSTICA / AEREO (22)
    "WEGE3": "INDUSTRIAL", "EMBR3": "INDUSTRIAL", "TUPY3": "INDUSTRIAL", "FRAS3": "INDUSTRIAL",
    "ROMI3": "INDUSTRIAL", "AERI3": "INDUSTRIAL", "RENT3": "INDUSTRIAL", "MOVI3": "INDUSTRIAL",
    "VAMO3": "INDUSTRIAL", "RAIL3": "INDUSTRIAL", "CCRO3": "INDUSTRIAL", "ECOR3": "INDUSTRIAL",
    "STBP3": "INDUSTRIAL", "PORT3": "INDUSTRIAL", "LOGN3": "INDUSTRIAL", "AZUL4": "INDUSTRIAL",
    "GOLL4": "INDUSTRIAL", "VLID3": "INDUSTRIAL", "TUTI3": "INDUSTRIAL", "SHUL4": "INDUSTRIAL",
    "GOAU3": "INDUSTRIAL", "RAPT4": "INDUSTRIAL",

    # TECNOLOGIA / COMUNICAÇÕES (12)
    "TOTS3": "TECNOLOGIA", "LWSA3": "TECNOLOGIA", "CASH3": "TECNOLOGIA", "POSI3": "TECNOLOGIA",
    "INTB3": "TECNOLOGIA", "NGRD3": "TECNOLOGIA", "IFCM3": "TECNOLOGIA", "VIVT3": "COMUNICAÇÕES",
    "TIMS3": "COMUNICAÇÕES", "DESK3": "TECNOLOGIA", "FIQE3": "TECNOLOGIA", "TELB4": "COMUNICAÇÕES",

    # IMOBILIÁRIO / FIIs (10)
    "ALSO3": "IMOBILIÁRIO", "MULT3": "IMOBILIÁRIO", "IGTI11": "IMOBILIÁRIO", "LOGG3": "IMOBILIÁRIO",
    "HGLG11": "IMOBILIÁRIO", "KNRI11": "IMOBILIÁRIO", "XPLG11": "IMOBILIÁRIO", "VISC11": "IMOBILIÁRIO",
    "HGRU11": "IMOBILIÁRIO", "MXRF11": "IMOBILIÁRIO"
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
NO_ENTRY_AFTER = "16:15"  # Fim das entradas (antes do fechamento nervoso)
CLOSE_ALL_BY = "16:45"  # FECHAMENTO FORÇADO (nunca posar no after)
DAILY_RESET_TIME = "10:15"  # Reset diário do circuit breaker

# ===========================
# GESTÃO DE RISCO
# ===========================
RISK_PER_TRADE_PCT = 0.006  # 1% da equity por trade padrão
REDUCED_RISK_PCT = 0.005  # 0.5% na sexta após 15h
MAX_DAILY_DRAWDOWN_PCT = 0.015  # Circuit breaker diário (2%)
ENABLE_NEWS_FILTER = True
NEWS_BLOCK_BEFORE_MIN = 30
NEWS_BLOCK_AFTER_MIN = 120
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
    "DEFAULT": 0.0020,  # 0.15% - realista para IOC em M15
}

MAX_TRADE_DURATION_CANDLES = 40  # Time-stop
# config.py - ADICIONAR
ADAPTIVE_THRESHOLDS = {
    "RISK_ON": {
        "min_signal_score": 30,  # Mais agressivo em bull
        "min_adx": 15,
        "min_volume_ratio": 1.2,
    },
    "RISK_OFF": {
        "min_signal_score": 40,  # Mais conservador em bear
        "min_adx": 22,
        "min_volume_ratio": 1.5,
    }
}
# ===========================
# PYRAMIDING
# ===========================
ENABLE_PYRAMID = True
PYRAMID_MAX_LEGS = 2
PYRAMID_ATR_DISTANCE = 1.0  # Segunda perna só após +1.0 ATR a favor
PYRAMID_RISK_SPLIT = [0.6, 0.4]
PYRAMID_REQUIREMENTS = {
    "min_adx": 35,  # ADX > 30 (tendência forte confirmada)
    "max_rsi_long": 65,  # RSI não sobrecomprado (compra)
    "min_rsi_short": 35,  # RSI não sobrevendido (venda)
    "volume_ratio": 1.3,  # Volume 20% acima da média
    "time_since_entry": 45,  # Mínimo 45 min desde primeira perna
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
        "adx_min": 20,
        "min_score": 40,
        "atr_max": 8.0,
        "min_volume_ratio": 1.1,     # Volume atual > 130% da média de 20 períodos
        "require_vwap_proximity": True,  # Preço perto do VWAP intraday (±1%)
        "min_momentum": 0.0007,  # Momentum mínimo mais exigente
    },
    "MID": {
        "start": "11:30",
        "end": "14:30",
        "adx_min": 18,
        "min_score": 35,
        "atr_max": 10.0,
        "min_volume_ratio": 1.05
    },
    "LATE": {
        "start": "14:30",
        "end": "16:55",
        "adx_min": 18,
        "min_score": 35,
        "atr_max": 12.0,
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
    "CURY3": {
        "ema_short": 18,
        "ema_long": 196,
        "rsi_low": 35,
        "rsi_high": 70,
        "adx_threshold": 17,
        "mom_min": 0.0007533133222969877
    },
    # Calmar: 5.41 | DD: 4.4% | Trades: 1

    "CSMG3": {
        "ema_short": 24,
        "ema_long": 175,
        "rsi_low": 35,
        "rsi_high": 67,
        "adx_threshold": 15,
        "mom_min": 0.001639577837804666
    },
    # Calmar: 4.16 | DD: 3.6% | Trades: 5

    "CAML3": {
        "ema_short": 24,
        "ema_long": 125,
        "rsi_low": 35,
        "rsi_high": 80,
        "adx_threshold": 15,
        "mom_min": 0.0016885776320875747
    },
    # Calmar: 4.13 | DD: 4.1% | Trades: 5

    "CEAB3": {
        "ema_short": 24,
        "ema_long": 183,
        "rsi_low": 35,
        "rsi_high": 66,
        "adx_threshold": 15,
        "mom_min": 0.000421082447848601
    },
    # Calmar: 4.07 | DD: 10.6% | Trades: 8

    "BBDC4": {
        "ema_short": 24,
        "ema_long": 154,
        "rsi_low": 35,
        "rsi_high": 79,
        "adx_threshold": 15,
        "mom_min": 0.00034755496073512657
    },
    # Calmar: 3.89 | DD: 2.5% | Trades: 5

    "CSNA3": {
        "ema_short": 17,
        "ema_long": 187,
        "rsi_low": 35,
        "rsi_high": 66,
        "adx_threshold": 15,
        "mom_min": 0.0005682824549681303
    },
    # Calmar: 3.75 | DD: 5.7% | Trades: 7

    "AURA33": {
        "ema_short": 18,
        "ema_long": 189,
        "rsi_low": 35,
        "rsi_high": 68,
        "adx_threshold": 15,
        "mom_min": 0.0010760457700169947
    },
    # Calmar: 3.54 | DD: 4.7% | Trades: 3

    "BPAN4": {
        "ema_short": 22,
        "ema_long": 195,
        "rsi_low": 35,
        "rsi_high": 75,
        "adx_threshold": 15,
        "mom_min": 0.0007713017955807511
    },
    # Calmar: 2.88 | DD: 7.7% | Trades: 6

    "SUZB3": {
        "ema_short": 22,
        "ema_long": 171,
        "rsi_low": 35,
        "rsi_high": 75,
        "adx_threshold": 15,
        "mom_min": 0.00038067696810886037
    },
    # Calmar: 2.75 | DD: 5.8% | Trades: 8

    "CSAN3": {
        "ema_short": 22,
        "ema_long": 195,
        "rsi_low": 35,
        "rsi_high": 75,
        "adx_threshold": 15,
        "mom_min": 0.0007713017955807511
    },
    # Calmar: 2.64 | DD: 3.1% | Trades: 2

    "FRAS3": {
        "ema_short": 25,
        "ema_long": 190,
        "rsi_low": 34,
        "rsi_high": 65,
        "adx_threshold": 19,
        "mom_min": 0.000791331558174043
    },
    # Calmar: 2.63 | DD: 3.4% | Trades: 1

    "MDNE3": {
        "ema_short": 22,
        "ema_long": 195,
        "rsi_low": 35,
        "rsi_high": 75,
        "adx_threshold": 15,
        "mom_min": 0.0007713017955807511
    },
    # Calmar: 2.46 | DD: 8.7% | Trades: 6

    "USIM3": {
        "ema_short": 18,
        "ema_long": 154,
        "rsi_low": 35,
        "rsi_high": 71,
        "adx_threshold": 15,
        "mom_min": 0.0005668058660073873
    },
    # Calmar: 2.32 | DD: 4.0% | Trades: 5

    "WEGE3": {
        "ema_short": 17,
        "ema_long": 138,
        "rsi_low": 35,
        "rsi_high": 78,
        "adx_threshold": 15,
        "mom_min": 0.001299977937021656
    },
    # Calmar: 2.30 | DD: 4.6% | Trades: 7

    "GOAU4": {
        "ema_short": 20,
        "ema_long": 133,
        "rsi_low": 35,
        "rsi_high": 82,
        "adx_threshold": 15,
        "mom_min": 0.0015392651830410152
    },
    # Calmar: 2.22 | DD: 5.4% | Trades: 3

}


LOW_LIQUIDITY_SYMBOLS = {}

# ============================================
# 🔥 PRIORIDADE 1 - ANTI-CHOP
# ============================================

# Cooldown após SL (evita reentrar em movimento lateral)
ANTI_CHOP = {
    "enabled": True,
    "cooldown_after_sl_minutes": 180,  # 30 min após SL antes de reentrar
    "min_range_pct": 1.2,  # Preço precisa andar ≥0.8% antes de nova entrada
    "max_consecutive_losses": 2,  # Máx 2 perdas seguidas → bloqueia ativo
    "block_duration_hours": 6,  # Bloqueia 6h após 2 perdas
    # ✨ NOVO: Cooldown progressivo por perda
    "progressive_cooldown": True,
    "cooldown_multipliers": {
        1: 1.0,   # 1ª perda: 2h (120 min × 1.0)
        2: 2.0,   # 2ª perda: 4h (120 min × 2.0)
        3: 4.0,   # 3ª perda: 8h (120 min × 4.0) - bloqueia resto do dia
    }
}

# ============================================
# 🔥 PRIORIDADE 2 - PIRÂMIDE INTELIGENTE
# ============================================

PYRAMID_REQUIREMENTS_ENHANCED = {
    **PYRAMID_REQUIREMENTS,  # Mantém configs antigas
    
    # ✅ NOVAS REGRAS CRÍTICAS
    "require_breakeven": True,  # SL no BE é OBRIGATÓRIO
    "require_1r_floating": True,  # OU ter +1R flutuante
    "min_time_between_legs_minutes": 15,  # Mín 15 min entre pernas
    "max_correlation_for_pyramid": 0.6,  # Não piramidar se correlação > 60%
}

# ============================================
# 🔥 PRIORIDADE 3 - LIMITE DIÁRIO POR ATIVO
# ============================================

DAILY_SYMBOL_LIMITS = {
    "enabled": True,
    "max_losing_trades_per_symbol": 1,  # Máx 2 perdas/ativo/dia
    "max_total_trades_per_symbol": 4,  # Máx 6 trades/ativo/dia (geral)
    "reset_time": "10:15",  # Reset junto com circuit breaker
}

# ============================================
# 🛡️ PROTEÇÃO ADICIONAL - RANGE MÍNIMO
# ============================================

MIN_PRICE_MOVEMENT = {
    "enabled": True,
    "min_atr_multiplier": 0.5,  # Preço deve ter movido ≥0.5 ATR
    "lookback_candles": 10,  # Verifica movimento nos últimos 10 candles
}
