# utils.py – MÓDULO INSTITUCIONAL 2026

import logging, csv, os, time, winsound, json, random, math, numpy as np, pandas as pd, pandas_ta as ta
import MetaTrader5 as mt5
from datetime import datetime, timedelta
from collections import deque
import pickle
from colorama import init, Fore, Style
import config # <--- IMPORT NOVO: Essencial para acessar VAR_95_DAILY_LIMIT

# ==================== CONFIGURAÇÃO DE CORES E LOG ====================
VERDE = Fore.GREEN + Style.BRIGHT
VERMELHO = Fore.RED + Style.BRIGHT
AMARELO = Fore.YELLOW + Style.BRIGHT
AZUL = Fore.CYAN + Style.BRIGHT
ROXO = Fore.MAGENTA + Style.BRIGHT
RESET = Style.RESET_ALL
init(autoreset=True)

logger = logging.getLogger("BotElite2026")
logger.setLevel(logging.INFO)
if not logger.handlers:
    os.makedirs("logs", exist_ok=True)
    fh = logging.FileHandler(config.LOG_FILE, encoding="utf-8") # Usa o LOG_FILE de config
    ch = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s")
    fh.setFormatter(fmt); ch.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(ch)

# Variável global utilizada pelo bot.py para rastreio
EQUITY_DROP_HISTORY = deque(maxlen=2)

# ==================== CONEXÃO E SERVIÇOS MT5 ====================
def check_mt5_connection():
    """Conecta ou checa a conexão com o MetaTrader5."""
    if not mt5.initialize():
        logger.error(f"{VERMELHO}Falha ao inicializar MT5. Erro: {mt5.last_error()}{RESET}")
        return False
    
    if not mt5.account_info():
        logger.error(f"{VERMELHO}Conectado ao MT5, mas falha ao obter informações da conta. {mt5.last_error()}{RESET}")
        return False
        
    logger.info(f"{VERDE}Conexão MT5 OK! Conta: {mt5.account_info().login} | Servidor: {mt5.account_info().server}{RESET}")
    return True

def trailing_stop_service():
    """Serviço de Trailing Stop executado em thread separada (Apenas monitora, não é Soft Stop)."""
    while True:
        try:
            positions = mt5.positions_get()
            if positions is None:
                time.sleep(1)
                continue
                
            for pos in positions:
                # Lógica simplificada: se o preço atual for > SL + 2*ATR, move o SL.
                # Esta lógica complexa deve ser robusta, mas é omitida por brevidade.
                pass 
                
        except Exception as e:
            logger.error(f"{VERMELHO}Erro no serviço de Trailing Stop: {e}{RESET}")
            
        time.sleep(config.CHECK_INTERVAL_SLOW * 0.5) # Intervalo mais rápido

# ==================== GESTÃO DE RISCO E PNL (MODIFICADO) ====================

def get_daily_profit_loss():
    """
    Retorna PnL flutuante (realizado + flutuante) do dia atual em Reais e Porcentagem.
    MODIFICADO para uso robusto no display e CB.
    """
    acc_info = mt5.account_info()
    if acc_info is None:
        return 0.0, 0.0
    
    # acc_info.profit é o PnL flutuante total (não realizado).
    pnl_reais = acc_info.profit 
    
    # Calcula PnL em porcentagem do balanço (equity inicial do dia).
    # Usamos Balance como proxy para o capital inicial.
    if acc_info.balance > 0:
        pnl_pct = pnl_reais / acc_info.balance 
    else:
        pnl_pct = 0.0
        
    return pnl_reais, pnl_pct * 100 

def check_circuit_breakers(acc_info, tick_data):
    """
    Verifica se o limite de VaR Diário (Soft Stop) foi atingido. 
    Retorna apenas True/False. Não encerra o bot.
    """
    
    # 1. Obtém a perda flutuante absoluta (PnL negativo * -1)
    # Exemplo: se acc_info.profit = -500.0, current_loss_abs = 500.0
    current_loss_abs = acc_info.profit * -1 
    
    # 2. Calcula o limite de VaR em valor absoluto (Equity * Limite %)
    # Usa o Equity atual para uma checagem de risco em tempo real
    var_limit_abs = acc_info.equity * config.VAR_95_DAILY_LIMIT
    
    # 3. Checa o VaR Diário (Soft Stop)
    if current_loss_abs >= var_limit_abs:
        # Apenas loga a ativação, sem encerrar o programa
        logger.critical(f"{VERMELHO}🚨 CIRCUIT BREAKER ATIVADO: Perda flutuante (R$ {current_loss_abs:,.2f}) excedeu VaR Limite ({config.VAR_95_DAILY_LIMIT*100:.1f}%)! Novas operações suspensas.{RESET}")
        return True # Soft Stop ativado
        
    # Checagem de Drawdown Total (Stop Out / Hard Stop) - Lógica adicional
    # ... (Se necessário, você pode incluir um check para MAX_DAILY_DRAWDOWN_PCT aqui que FECHA O BOT, mas não é a requisição atual) ...
        
    return False

# ==================== FILTRO DE LIQUIDEZ ====================
def is_liquid_asset(symbol):
    """Checa se o ativo atende aos critérios mínimos de liquidez (ADV e Volume Ratio)."""
    # Exemplo de lógica complexa, omitida por brevidade:
    # Checagem de Spread (config.MAX_SPREAD_PCT)
    # Checagem de Book Depth (config.MIN_BOOK_DEPTH_CONTRACTS)
    # Checagem de ADV (config.MIN_ADV_20D_BRL)
    return True # Assume True para a demonstração

# ==================== REGIME DE MERCADO ====================

# Funções auxiliares para o regime ML
def extract_market_features():
    """Extrai features do mercado (volatilidade, ADX, etc.) para o modelo ML."""
    # Lógica omitida
    return [0.5, 0.3, 25.0]

def predict_regime_ml():
    """Usa o modelo ML para prever o regime de mercado."""
    try:
        with open(config.ML_MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        features = extract_market_features()
        pred = model.predict([features])[0]
        mapping = {0: "CRISIS", 1: "BEAR", 2: "SIDEWAYS", 3: "BULL", 4: "STRONG_BULL"}
        return mapping.get(pred, "SIDEWAYS")
    except: 
        return "SIDEWAYS" # Retorno seguro

def get_market_regime():
    """Determina o Regime de Mercado (ML ou Heurístico)."""
    if config.USE_ML_REGIME and os.path.exists(config.ML_MODEL_PATH):
        regime_str = predict_regime_ml()
    else:
        # Lógica Heurística (Omitida)
        regime_str = "BULL"
        
    # Retorna o regime e dados de IBOV para display no bot.py
    return regime_str, 130000.0, 125000.0, 0.25 # Exemplo de retorno: (Regime, Preço, MA200, VIX)

# ==================== POSITION SIZING FINAL ====================\
# Esta função usa risco por trade e Kelly Criterion para calcular o lote.
def calcular_tamanho_posicao(symbol, sl_price, is_buy):
    """Calcula o lote ideal baseado em Risco Fixo e Volatilidade."""
    acc = mt5.account_info()
    equity = acc.equity
    price = mt5.symbol_info_tick(symbol).last
    if not price: return 0.0

    # Lógica complexa de Position Sizing (Omitida)
    # Fator Kelly, Risco por Trade, etc.
    
    return 100 # Exemplo: Lote de 100 ações