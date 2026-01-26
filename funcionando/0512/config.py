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
    "PETR4":"ENERGY", "PETR3":"ENERGY", "PRIO3":"ENERGY", "RRRP3":"ENERGY",
    "VALE3":"MATERIALS", "GGBR4":"MATERIALS", "CSNA3":"MATERIALS", "USIM5":"MATERIALS",
    "ITUB4":"FINANCIALS", "BBDC4":"FINANCIALS", "BBAS3":"FINANCIALS", "B3SA3":"FINANCIALS",
    "ABEV3":"STAPLES", "JBSS3":"STAPLES", "BRFS3":"STAPLES",
    "WEGE3":"INDUSTRIALS", "RAIL3":"INDUSTRIALS", "RENT3":"INDUSTRIALS",
    "MGLU3":"CDISCRETIONARY", "LREN3":"CDISCRETIONARY",
    "EQTL3":"UTILITIES", "SBSP3":"UTILITIES", "CPFE3":"UTILITIES",
    "HAPV3":"HEALTH", "RADL3":"HEALTH"
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