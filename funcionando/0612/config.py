# config.py – BOT ELITE 2026 PRO (VERSÃO INSTITUCIONAL)
import datetime
import MetaTrader5 as mt5
import os
import json 
import logging

# ==================== GERAIS ====================
MODE = "REAL" # Alterar para REAL quando pronto
CORRETORA_SUFFIX = "" # Ex: ".SA"
LOG_FILE = "logs/bot_elite_pro.log"

# ==================== HORÁRIO DE OPERAÇÃO ====================
START_TIME = datetime.time(10, 15) # Aguarda abertura do mercado à vista e estabilização
END_TIME = datetime.time(16, 45)   # Fecha antes do leilão final
CHECK_INTERVAL_SLOW = 30 
SCREEN_UPDATE_INTERVAL = 5 

# ==================== GESTÃO DE RISCO (INSTITUCIONAL) ====================
MAX_SPREAD_CENTS = 0.08 
RISCO_POR_TRADE_PCT = 0.50  # 0.5% do Equity total em risco por operação (Padrão conservador)
ALAVANCAGEM_MAXIMA = 3.0    # Trava de segurança: Não alocar mais que 3x o patrimônio (Notional)
MAX_ATIVOS_ABERTOS = 5      
DAILY_LOSS_LIMIT_PCT = -2.0 # Stop Loss Global do Dia

# ATR & Volatilidade
ATR_PERIOD = 14
ATR_MULTIPLIER_SL = 2.0     # Stop Loss técnico
TAKE_PROFIT_MULTIPLIER = 1.5 # Risco/Retorno de 1:1.5 (TP é 1.5x o tamanho do SL)

# Controle de Setor
MAX_EXPOSURE_PER_SECTOR_PCT = 0.25 # Máximo 25% do capital alocado em um único setor

# ==================== FILTROS TÉCNICOS AVANÇADOS ====================
# Indicadores
TIMEFRAME_MT5 = mt5.TIMEFRAME_M5
RSI_PERIOD = 14
VOLUME_MA_PERIOD = 20

# Filtro de Tendência (ADX) - Evita operar em lateralização
ADX_PERIOD = 14
ADX_MIN_THRESHOLD = 20 # Só opera se ADX > 20 (Mercado com tendência)

# Filtro de Regime de Mercado (MM200)
USE_MARKET_REGIME_FILTER = True # Se True, só compra se IBOV > MM200
IBOV_MA_PERIOD = 200

# ==================== OTIMIZAÇÃO E ARQUIVOS ====================
PARAMETROS_FILE_COMPRA = 'data/optimized_params_compra.json'
PARAMETROS_FILE_VENDA = 'data/optimized_params_venda.json'
CSV_FILE = "trades/trades_historico.csv"
EWMA_DECAY_FACTOR = 0.98

# Variáveis Dinâmicas (Defaults seguros caso otimização falhe)
EMA_FAST_DEFAULT = 9
EMA_SLOW_DEFAULT = 21

PARAMETROS_OTIMIZADOS_COMPRA = {
    "data": "DEFAULT", "score": 0.0, "ema_fast": EMA_FAST_DEFAULT, "ema_slow": EMA_SLOW_DEFAULT, 
    "rsi_max": 70, "momentum_min": 0.2, "volume_mult": 1.2, "winrate": 0.0
}
PARAMETROS_OTIMIZADOS_VENDA = {
    "data": "DEFAULT", "score": 0.0, "ema_fast": EMA_FAST_DEFAULT, "ema_slow": EMA_SLOW_DEFAULT, 
    "rsi_min": 30, "momentum_max_neg": -0.2, "volume_mult": 1.2, "winrate": 0.0
}

CANDIDATOS_DINAMICOS = []
MAX_ASSETS_SCANEADOS = 80
MAX_ASSETS_NA_TELA = 12
SCREEN_UPDATE_REQUIRED = True

# Índice de Referência (Automático no utils, mas definimos o fallback aqui)
IBOV_SYMBOL_FALLBACK = "WIN$N" # Símbolo genérico do Profit/MT5 (ajustado dinamicamente)
IBOV_MAX_DROP_PCT = -1.5 

# ==================== LISTA DE ATIVOS (Liquidez > 50M/dia) ====================
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


def load_optimized_params(logger: logging.Logger):
    """Carrega parâmetros e valida integridade."""
    global PARAMETROS_OTIMIZADOS_COMPRA, PARAMETROS_OTIMIZADOS_VENDA
    
    for nome_arq, dict_params, label in [
        (PARAMETROS_FILE_COMPRA, PARAMETROS_OTIMIZADOS_COMPRA, "Compra"),
        (PARAMETROS_FILE_VENDA, PARAMETROS_OTIMIZADOS_VENDA, "Venda")
    ]:
        if os.path.exists(nome_arq):
            try:
                with open(nome_arq, 'r') as f:
                    dados = json.load(f)
                    # Validação básica de chaves
                    if 'ema_fast' in dados:
                        dict_params.update(dados)
                        logger.info(f"Params {label} carregados. Score: {dados.get('score', 0):.1f}")
            except Exception as e:
                logger.error(f"Erro ao ler {nome_arq}: {e}")