# config.py – BOT ELITE 2026 PRO (INSTITUCIONAL KERNEL)
import datetime
import os
import json 
import logging

# ==================== GERAIS ====================
MODE = "REAL"  # 'REAL' ou 'SIMULATION'
LOG_FILE = "logs/bot_elite_institutional.log"
TRADES_FILE = "trades/trades_historico.csv"

# ==================== HORÁRIO DE OPERAÇÃO ====================
START_TIME = datetime.time(10, 15) 
END_TIME = datetime.time(16, 45)   
CHECK_INTERVAL_SLOW = 15  # Loop principal

# ==================== GESTÃO DE RISCO (KELLY & VOL TARGET) ====================
USE_KELLY_CRITERION = True
KELLY_FRACTION = 0.4        # Fração de Kelly (Conservador: 0.3 a 0.5)
MAX_RISK_PER_TRADE = 0.70   # Cap absoluto de risco por trade (% do Equity)
TARGET_VOL_ANNUAL = 0.20    # Alvo de volatilidade anualizada (20%)
MAX_LEVERAGE = 3.0          # Alavancagem nocional máxima

# Filtro de Pânico (IV Check)
PANIC_IV_THRESHOLD = 70.0   # Se Volatilidade Anualizada do Índice > 70%
PANIC_RISK_REDUCTION = 0.15 # Risco cai para 0.15% em pânico

# Circuit Breakers Globais
MAX_DAILY_DRAWDOWN = -3.5   # Encerra o dia se atingir -3.5%
PEAK_EQUITY_STOP = 0.92     # Encerra TUDO se equity cair para 92% do pico histórico

# ==================== EXECUÇÃO (TWAP/ICEBERG) ====================
USE_ICEBERG_EXECUTION = True
ICEBERG_SPLIT = 4           # Dividir ordem em 4 pernas
ICEBERG_MIN_DELAY = 4       # Segundos min entre pernas
ICEBERG_MAX_DELAY = 18      # Segundos max entre pernas
ORDER_TIMEOUT_SEC = 60      # Tempo limite para Limit Order virar Market

# ==================== PARAMETRIZAÇÃO DINÂMICA (REGIMES) ====================
# Arquivos de parâmetros por regime
PARAMS_BULL = 'data/params_bull.json'
PARAMS_BEAR = 'data/params_bear.json'
PARAMS_SIDE = 'data/params_sideways.json'

# Definição de Regime
IBOV_SYMBOL = "WIN$N"     # Ajuste para o contrato vigente ou 
IBOV_MM_PERIOD = 200      # Média Móvel Regimeíndice
IBOV_ADX_SIDEWAYS = 18    # Abaixo disso é lateralidade

# ==================== SETUP TÉCNICO ====================
TIMEFRAME_MT5 = 5 # M5 (passado como int para utils se necessário, ou constante MT5 no main)
MIN_SCORE_ENTRY = 85.0 # Score mínimo (0-100) para entrar

# Defaults de Segurança
DEFAULT_PARAMS = {
    "ema_fast": 9, "ema_slow": 21, "rsi_period": 14, 
    "adx_period": 14, "tp_mult": 1.5, "sl_atr_mult": 2.0
}

CANDIDATOS_BASE = [
    "PETR4", "VALE3", "ITUB4", "BBDC4", "BBAS3", "PETR3", "PRIO3", "WEGE3",
    "GGBR4", "CSNA3", "USIM5", "ABEV3", "LREN3", "MGLU3", "VIVT3", "SUZB3",
    "RENT3", "HAPV3", "RADL3", "EQTL3", "B3SA3", "JBSS3", "RAIL3", "SBSP3"
]

# Setores (Mapeamento simplificado)
SETORES = {
    # ========================= ENERGIA & UTILIDADES (ENERGY & UTILITIES) =========================
    "PETR4":"ENERGY", "PETR3":"ENERGY", "PRIO3":"ENERGY", "RRRP3":"ENERGY",
    "ENAT3":"ENERGY", "UGPA3":"ENERGY", "CCRO3":"UTILITIES", "CSMG3":"UTILITIES",
    "EQTL3":"UTILITIES", "SBSP3":"UTILITIES", "CPFE3":"UTILITIES", "ENGI11":"UTILITIES",
    "ALUP11":"UTILITIES", "TAEE11":"UTILITIES", "CMIG4":"UTILITIES", "ELET3":"UTILITIES",
    "ELET6":"UTILITIES", "COCE5":"UTILITIES", "CESP6":"UTILITIES", "CPLE6":"UTILITIES",
    "AFLT3":"UTILITIES", "CLSC4":"UTILITIES", "RNEW11":"UTILITIES", "TRPL4":"UTILITIES",
    "AESB3":"UTILITIES", "GEPA4":"UTILITIES", "POMO4":"INDUSTRIALS",
    
    # ========================= FINANCEIROS (FINANCIALS) =========================
    "ITUB4":"FINANCIALS", "ITUB3":"FINANCIALS", "BBDC4":"FINANCIALS", "BBDC3":"FINANCIALS", 
    "BBAS3":"FINANCIALS", "SANB11":"FINANCIALS", "BPAC11":"FINANCIALS", "B3SA3":"FINANCIALS",
    "CXSE3":"FINANCIALS", "BBSE3":"FINANCIALS", "IRBR3":"FINANCIALS", "PSSA3":"FINANCIALS",
    "VIVA3":"FINANCIALS", "AMER3":"FINANCIALS", "CIEL3":"FINANCIALS", "MERC3":"FINANCIALS",
    "MGLU3":"CDISCRETIONARY", "PINE4":"FINANCIALS", "BMGB4":"FINANCIALS", "BSLI4":"FINANCIALS",
    "PATI3":"FINANCIALS", "MODL11":"FINANCIALS", "JPSA3":"FINANCIALS", "SEQL3":"FINANCIALS",
    "GETT3":"FINANCIALS", "GETT4":"FINANCIALS",
    
    # ========================= MATERIAIS BÁSICOS (MATERIALS) =========================
    "VALE3":"MATERIALS", "GGBR4":"MATERIALS", "CSNA3":"MATERIALS", "USIM5":"MATERIALS",
    "SUZB3":"MATERIALS", "KLAB3":"MATERIALS", "GOAU4":"MATERIALS", "BRKM5":"MATERIALS",
    "BERK3":"MATERIALS", "FRAS3":"MATERIALS", "HAGA4":"MATERIALS", "ROMI3":"MATERIALS", 
    "TUPY3":"MATERIALS", "UNIP6":"MATERIALS", "VULC3":"MATERIALS", "CMIN3":"MATERIALS",
    "APER3":"MATERIALS", "FESA4":"MATERIALS", "PATI3":"MATERIALS", "MILS3":"MATERIALS",
    "TGMA3":"MATERIALS", "SGPS3":"MATERIALS", "CRIV4":"MATERIALS", "GSHP3":"MATERIALS",
    "TECN3":"MATERIALS", "TEND3":"MATERIALS", "VAMO3":"MATERIALS",
    
    # ========================= BENS DE CONSUMO CÍCLICO (CDISCRETIONARY) =========================
    "MGLU3":"CDISCRETIONARY", "LREN3":"CDISCRETIONARY", "CVCB3":"CDISCRETIONARY", 
    "AMAR3":"CDISCRETIONARY", "SOMA3":"CDISCRETIONARY", "AREZ3":"CDISCRETIONARY", 
    "PETZ3":"CDISCRETIONARY", "AZUL4":"CDISCRETIONARY", "GOLL4":"CDISCRETIONARY",
    "ANIM3":"CDISCRETIONARY", "YDUQ3":"CDISCRETIONARY", "CASH3":"CDISCRETIONARY",
    "C&A":"CDISCRETIONARY", "VIVR3":"CDISCRETIONARY", "CRPG6":"CDISCRETIONARY",
    "ELMD3":"CDISCRETIONARY", "ESPA3":"CDISCRETIONARY", "LAME4":"CDISCRETIONARY",
    "SLED3":"CDISCRETIONARY", "LCAM3":"CDISCRETIONARY", "ALPA4":"CDISCRETIONARY",
    "TFCO4":"CDISCRETIONARY", "BRPR3":"CDISCRETIONARY", "PDGR3":"CDISCRETIONARY",
    
    # ========================= BENS DE CONSUMO NÃO CÍCLICO (STAPLES) =========================
    "ABEV3":"STAPLES", "JBSS3":"STAPLES", "BRFS3":"STAPLES", "BEEF3":"STAPLES",
    "MDIA3":"STAPLES", "CAML3":"STAPLES", "HAPV3":"HEALTH", "RADL3":"HEALTH",
    "FLRY3":"HEALTH", "DASA3":"HEALTH", "ODPV3":"HEALTH", "PARD3":"HEALTH",
    "PMET3":"HEALTH", "PNVL3":"HEALTH", "RDOR3":"HEALTH", "FRAS3":"STAPLES",
    "HYPE3":"STAPLES", "LIGT3":"STAPLES", "MTRE3":"STAPLES", "NTCO3":"STAPLES",
    "PRIO3":"ENERGY", "RRRP3":"ENERGY", "SGPS3":"STAPLES", "TGMA3":"MATERIALS",
    "VULC3":"MATERIALS", "VAMO3":"INDUSTRIALS", "COGN3":"CDISCRETIONARY", "PCAR3":"STAPLES",
    
    # ========================= SAÚDE (HEALTHCARE) =========================
    "HAPV3":"HEALTHCARE", "QUAL3":"HEALTHCARE", "RDOR3":"HEALTHCARE", "FLRY3":"HEALTHCARE",
    "GMAT3":"HEALTHCARE", "FLRY3":"HEALTHCARE", "DASA3":"HEALTHCARE", "ODPV3":"HEALTHCARE",
    "PARD3":"HEALTHCARE", "PNVL3":"HEALTHCARE", "MILS3":"HEALTHCARE",
    
    # ========================= INDÚSTRIA (INDUSTRIALS) =========================
    "WEGE3":"INDUSTRIALS", "RAIL3":"INDUSTRIALS", "EMBR3":"INDUSTRIALS", "AZUL4":"INDUSTRIALS",
    "GOLL4":"INDUSTRIALS", "TASA4":"INDUSTRIALS", "GRND3":"INDUSTRIALS", "ECOR3":"INDUSTRIALS",
    "LOGG3":"INDUSTRIALS", "NTCO3":"INDUSTRIALS", "SLCE3":"INDUSTRIALS", "TTEN3":"INDUSTRIALS",
    "VIVR3":"CDISCRETIONARY", "YDUQ3":"CDISCRETIONARY", "EVEN3":"REALESTATE", 
    "EZTC3":"REALESTATE", "CYRE3":"REALESTATE", "MRVE3":"REALESTATE", "TRIS3":"REALESTATE",
    "TEKA3":"INDUSTRIALS", "VIAL3":"CDISCRETIONARY", "TUPY3":"MATERIALS", "LEVE3":"INDUSTRIALS",
    
    # ========================= TECNOLOGIA E TELECOM (TECHNOLOGY & COMMUNICATION) =========================
    "TOTS3":"TECHNOLOGY", "POSI3":"TECHNOLOGY", "VIVT3":"COMMUNICATION", "WIZC3":"FINANCIALS",
    "LWSA3":"TECHNOLOGY", "INTB3":"TECHNOLOGY", "TECN3":"TECHNOLOGY", "BEEF3":"STAPLES",
    "JHSF3":"REALESTATE", "AALR3":"HEALTHCARE", "AZEV4":"CDISCRETIONARY", "AURA33":"MATERIALS",
    "BRIT3":"REALESTATE", "CPLE6":"UTILITIES", "CRFB3":"STAPLES", "CTSA4":"UTILITIES",
    "DEXP3":"UTILITIES", "DIRR3":"REALESTATE", "HBSA3":"REALESTATE", "IGTI11":"CDISCRETIONARY",
    "JSLG3":"INDUSTRIALS", "MOVI3":"INDUSTRIALS", "PLPL3":"HEALTHCARE", "SMAL11":"FINANCIALS",
    
    # ========================= IMOBILIÁRIO (REAL ESTATE) =========================
    "MRVE3":"REALESTATE", "EZTC3":"REALESTATE", "CYRE3":"REALESTATE", "TRIS3":"REALESTATE",
    "EVEN3":"REALESTATE", "JHSF3":"REALESTATE", "LIGT3":"UTILITIES", "PARD3":"HEALTHCARE",
    "PDGR3":"REALESTATE", "TEND3":"REALESTATE", "TEXP3":"REALESTATE", "BRIT3":"REALESTATE",
    "DIRR3":"REALESTATE", "HBOR3":"REALESTATE", "MULT3":"CDISCRETIONARY", "LOGN3":"INDUSTRIALS",
}
