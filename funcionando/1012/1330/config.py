# config.py – BOT ELITE 2026 PRO – VERSÃO INSTITUCIONAL DEFINITIVA
import datetime
import os
import MetaTrader5 as mt5

# ==================== GERAIS ====================
MODE = "REAL" # Pode ser "DEMO" ou "REAL"
LOG_FILE = "logs/bot_elite_2026.log"
TRADES_FILE = "trades/trades_2026.csv"
METRICS_FILE = "metrics/live_metrics.json"
OPTIMIZER_HISTORY_FILE = "data/optimizer_history.json"

# ==================== HORÁRIO DE OPERAÇÃO ====================
TIMEFRAME_MT5 = mt5.TIMEFRAME_M5
START_TIME = datetime.time(10, 10)  # Início real das operações
END_TIME = datetime.time(16, 45)   # Fim real das operações
CHECK_INTERVAL_SLOW = 8            # Intervalo de loop em segundos

# ==================== REGIME DE MERCADO ====================
USE_ML_REGIME = True
ML_MODEL_PATH = "models/regime_xgboost_2025.pkl"
IBOV_SYMBOL = "WINZ25" # Índice Bovespa Futuro para checagem do regime

# ==================== FILTRO DE LIQUIDEZ ====================
MIN_ADV_20D_BRL = 8_000_000         # Volume Diário Médio Mínimo (20 dias)
MIN_VOLUME_5MIN_RATIO = 0.018       # Volume 5 min vs ADV (Filtro de Spike)
MAX_SPREAD_PCT = 0.16               # Spread máximo permitido em % do preço
MIN_BOOK_DEPTH_CONTRACTS = 600      # Lotes mínimos no book de ofertas

# ==================== GESTÃO DE RISCO AVANÇADA (VaR & Circuit Breaker) ====================
MAX_TOTAL_DRAWDOWN_PCT = 0.12       # 12.0% - Limite Histórico
MAX_EXPOSURE_PCT = 0.45             # Máxima exposição total em % do Equity
MAX_POSITIONS = 10                  # Máximo de posições abertas
MAX_POSITIONS_PER_SECTOR = 2        # Máximo de posições por setor (Diversificação)
COMISSAO_POR_LOTE = 0.05            # R$ 0.05 por lote (Custo)
SLIPPAGE_TICKS_MEDIO = 2.6          # 2.6 Ticks médios de slippage (0.01/tick)
TARGET_VOL_ANNUAL = 0.20
MAX_DAILY_DRAWDOWN_PERCENT = 3.0    # Drawdown MÁXIMO diário aceitável (3.0% do Equity Inicial)
CLOSE_ON_DRAWDOWN_FACTOR = 0.5      # Volatilidade Anual Desejada

# NOVO/MODIFICADO: Limite de VaR para o Soft Stop
# Se a perda flutuante (PnL negativo) atingir este % do Equity, novas operações são suspensas.
VAR_95_DAILY_LIMIT = 0.025 
MAX_TRADES_PER_CYCLE = 2        # 3.8% de perda flutuante (Circuit Breaker Soft Stop)

# ==================== OTIMIZAÇÃO (WALK-FORWARD) ====================
# Caminhos para os parâmetros otimizados que o bot irá carregar
PARAMS_STRONG_BULL = "params_strong_bull.json"
PARAMS_BULL = "params_bull.json"
PARAMS_SIDEWAYS = "params_sideways.json"
PARAMS_BEAR = "params_bear.json"
PARAMS_CRISIS = "params_crisis.json"
OPTIMIZER_HISTORY_FILE = "optimizer_history.json"

# Parâmetros default caso o arquivo de otimização falhe
DEFAULT_PARAMS = {
    "regime": "DEFAULT",
    "sharpe_medio": 0.0,
    "side": "COMPRA",  # <-- CORREÇÃO PRINCIPAL: Defina o lado padrão
    "ema_fast": 12,
    "ema_slow": 26,
    "rsi_level": 60,
    "momentum_min_pct": 0.4,
    "adx_min": 25,
    "sl_atr_mult": 2.0,
    "tp_atr_mult": 4.0,
    "data": "N/A"
}

# Arquivos de parâmetros adaptativos (você pode ter que adicionar esta seção também)
PARAMS_STRONG_BULL = "params_strong_bull.json"
PARAMS_BULL = "params_bull.json"
PARAMS_SIDEWAYS = "params_sideways.json"
PARAMS_BEAR = "params_bear.json"
PARAMS_CRISIS = "params_crisis.json"


# ==================== UNIVERSO DE ATIVOS B3 (EXEMPLO) ====================
# Utilizado para o Scan de Oportunidades no bot.py
SYMBOL_MAP = {
    # 1. FINANCEIRO & SEGUROS (20)
    "ITUB4":"FINANCIALS", "BBDC4":"FINANCIALS", "BBAS3":"FINANCIALS", "SANB11":"FINANCIALS", 
    "B3SA3":"FINANCIALS", "BPAC11":"FINANCIALS", "CXSE3":"FINANCIALS", "BPAN4":"FINANCIALS",
    "BTG11":"FINANCIALS", "CIEL3":"FINANCIALS", "PSSA3":"FINANCIALS", "SULA11":"FINANCIALS", 
    "IRBR3":"FINANCIALS", "MERC3":"FINANCIALS", "BRSR6":"FINANCIALS", "BBSA3":"FINANCIALS",
    "WIZC3":"FINANCIALS", "SEQL3":"CDISCRETIONARY", "ODPV3":"CDISCRETIONARY", "CASH3":"CDISCRETIONARY", 
    
    # 2. COMMODITIES & MATERIAIS BÁSICOS (16)
    "VALE3":"MATERIALS", "CSNA3":"MATERIALS", "GGBR4":"MATERIALS", "USIM5":"MATERIALS", 
    "BRKM5":"MATERIALS", "KLBN11":"MATERIALS", "SUZB3":"MATERIALS", "BRAP4":"MATERIALS",
    "CMIN3":"MATERIALS", "GOAU4":"MATERIALS", "FESA4":"MATERIALS", "TGMA3":"INDUSTRIALS",
    "SLCE3":"STAPLES", "JBSS3":"STAPLES", "AZUL4":"INDUSTRIALS", "GOLL4":"INDUSTRIALS",

    # 3. ENERGIA & UTILITIES (16)
    "PETR4":"ENERGY", "PETR3":"ENERGY", "PRIO3":"ENERGY", "ENEV3":"ENERGY", 
    "ELET3":"UTILITIES", "CMIG4":"UTILITIES", "EQTL3":"UTILITIES", "CPFE3":"UTILITIES", 
    "TAEE11":"UTILITIES", "SABESP3":"UTILITIES", "ENBR3":"UTILITIES", "RECV3":"ENERGY",
    "RRRP3":"ENERGY", "ENAT3":"ENERGY", "CSAN3":"ENERGY", "ALUP11":"UTILITIES", 
    
    # 4. CONSUMO, VAREJO & INDÚSTRIA (24)
    "ABEV3":"STAPLES", "ASAI3":"STAPLES", "BEEF3":"STAPLES", "MRFG3":"STAPLES", "MDIA3":"STAPLES",
    "LREN3":"CDISCRETIONARY", "MGLU3":"CDISCRETIONARY", "ARZZ3":"CDISCRETIONARY", 
    "CVCB3":"CDISCRETIONARY", "SOMA3":"CDISCRETIONARY", "ALPA4":"CDISCRETIONARY",
    "YDUQ3":"CDISCRETIONARY", "COGN3":"CDISCRETIONARY", "PETZ3":"CDISCRETIONARY",
    "VVAR3":"CDISCRETIONARY", "RENT3":"INDUSTRIALS", "WEGE3":"INDUSTRIALS", "RAIL3":"INDUSTRIALS",
    "EMBR3":"INDUSTRIALS", "TEND3":"CDISCRETIONARY", "JALL3":"STAPLES", "LCAM3":"INDUSTRIALS",
    "LOGG3":"INDUSTRIALS", "MRVE3":"REALESTATE",

    # 5. SAÚDE, TECNOLOGIA & IMOBILIÁRIO (24)
    "RADL3":"HEALTHCARE", "AALR3":"HEALTHCARE", "PARD3":"HEALTHCARE", "QUAL3":"HEALTHCARE",
    "HYPE3":"HEALTHCARE", "RDOR3":"HEALTHCARE", "FLRY3":"HEALTHCARE", "TOTS3":"TECHNOLOGY", 
    "LWSA3":"TECHNOLOGY", "INTB3":"TECHNOLOGY", "VIVT3":"COMMUNICATION", "TIMS3":"COMMUNICATION", 
    "MYPK3":"INDUSTRIALS", "ANIM3":"CDISCRETIONARY", "POSI3":"TECHNOLOGY", "CYRE3":"REALESTATE", 
    "EZTC3":"REALESTATE", "JHSF3":"REALESTATE", "DIRR3":"REALESTATE", "EVEN3":"REALESTATE",
    "CAML3":"STAPLES", "GFSA3":"REALESTATE", "MOVI3":"INDUSTRIALS", "JPSA3":"CDISCRETIONARY"
}

MAX_PER_SECTOR = 2        # Nunca mais de 2 posições por setor
MAX_POSITIONS_TOTAL = 10  # Limite absoluto da mesa