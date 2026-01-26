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
MAX_DAILY_DRAWDOWN_PCT = 0.04       # 4.0% - Limite de Drawdown (Stop Out)
MAX_TOTAL_DRAWDOWN_PCT = 0.12       # 12.0% - Limite Histórico
MAX_EXPOSURE_PCT = 0.45             # Máxima exposição total em % do Equity
MAX_POSITIONS = 15                  # Máximo de posições abertas
MAX_POSITIONS_PER_SECTOR = 3        # Máximo de posições por setor (Diversificação)
COMISSAO_POR_LOTE = 0.05            # R$ 0.05 por lote (Custo)
SLIPPAGE_TICKS_MEDIO = 2.6          # 2.6 Ticks médios de slippage (0.01/tick)
TARGET_VOL_ANNUAL = 0.20            # Volatilidade Anual Desejada

# NOVO/MODIFICADO: Limite de VaR para o Soft Stop
# Se a perda flutuante (PnL negativo) atingir este % do Equity, novas operações são suspensas.
VAR_95_DAILY_LIMIT = 0.038          # 3.8% de perda flutuante (Circuit Breaker Soft Stop)

# ==================== OTIMIZAÇÃO (WALK-FORWARD) ====================
# Caminhos para os parâmetros otimizados que o bot irá carregar
PARAMS_STRONG_BULL = "data/params_strong_bull.json"
PARAMS_BULL = "data/params_bull.json"
PARAMS_SIDEWAYS = "data/params_sideways.json"
PARAMS_BEAR = "data/params_bear.json"
PARAMS_CRISIS = "data/params_crisis.json"
OPTIMIZER_HISTORY_FILE = "data/optimizer_history.json"

# Parâmetros default caso o arquivo de otimização falhe
DEFAULT_PARAMS = {
    "ema_fast": 12, "ema_slow": 30, "rsi_level": 70, 
    "momentum_min": 0.5, "adx_min": 25, 
    "sl_atr_mult": 2.5, "tp_mult": 1.5, "side": "COMPRA"
}

# ==================== UNIVERSO DE ATIVOS B3 (EXEMPLO) ====================
# Utilizado para o Scan de Oportunidades no bot.py
UNIVERSE_B3 = {
    # Commodities/Materiais
    "PETR4":"ENERGY", "VALE3":"MATERIALS", "CSNA3":"MATERIALS", "GGBR4":"MATERIALS", "USIM5":"MATERIALS",
    "BRKM5":"MATERIALS", "CMIG4":"UTILITIES", "KLBN11":"MATERIALS", "SUZB3":"MATERIALS",
    
    # Financeiro
    "ITUB4":"FINANCIALS", "BBDC4":"FINANCIALS", "BBAS3":"FINANCIALS", "SANB11":"FINANCIALS", 
    "B3SA3":"FINANCIALS", "BPAC11":"FINANCIALS", "CXSE3":"FINANCIALS", "ASAI3":"STAPLES",
    
    # Consumo Cíclico e Não Cíclico
    "WEGE3":"INDUSTRIALS", "RENT3":"INDUSTRIALS", "LREN3":"CDISCRETIONARY", "MGLU3":"CDISCRETIONARY",
    "RAIL3":"INDUSTRIALS", "HAPV3":"HEALTHCARE", "ENEV3":"ENERGY", "EQTL3":"UTILITIES",
    "PRIO3":"ENERGY", "TGMA3":"INDUSTRIALS", "NTCO3":"INDUSTRIALS", "SLCE3":"INDUSTRIALS", 
    
    # Outros
    "TOTS3":"TECHNOLOGY", "POSI3":"TECHNOLOGY", "VIVT3":"COMMUNICATION", "WIZC3":"FINANCIALS",
    "LWSA3":"TECHNOLOGY", "INTB3":"TECHNOLOGY", "TECN3":"TECHNOLOGY", "BEEF3":"STAPLES",
    "JHSF3":"REALESTATE", "AALR3":"HEALTHCARE", 
    
    # Expansão
    "CPFE3":"UTILITIES", "COGN3":"CDISCRETIONARY", "AZUL4":"INDUSTRIALS", "CVCB3":"CDISCRETIONARY",
    "GOLL4":"INDUSTRIALS", "SOMA3":"CDISCRETIONARY", "ARZZ3":"CDISCRETIONARY", "MRFG3":"STAPLES",
    "BPAN4":"FINANCIALS", "ODPV3":"CDISCRETIONARY", "ALPA4":"CDISCRETIONARY",
    "YDUQ3":"CDISCRETIONARY", "EZTC3":"REALESTATE", "CYRE3":"REALESTATE", "MRVE3":"REALESTATE",
}