# utils.py – VERSÃO FINAL ELITE PRO (COM ADX E RISCO INSTITUCIONAL)
import logging
import csv
import os
import time
import winsound
import numpy as np
import pandas as pd
import pandas_ta as ta
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Optional, Tuple
import MetaTrader5 as mt5
import config
from colorama import init, Fore, Style

init(autoreset=True)

# Cores
VERDE = Fore.GREEN + Style.BRIGHT
VERMELHO = Fore.RED + Style.BRIGHT
AMARELO = Fore.YELLOW + Style.BRIGHT
AZUL = Fore.CYAN + Style.BRIGHT
ROXO = Fore.MAGENTA + Style.BRIGHT
BRANCO = Fore.WHITE + Style.BRIGHT
RESET = Style.RESET_ALL

def setup_logger():
    logger = logging.getLogger("BotElitePro")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        os.makedirs("logs", exist_ok=True)
        fh = logging.FileHandler(config.LOG_FILE, encoding="utf-8")
        ch = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s")
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger

logger = setup_logger()
config.load_optimized_params(logger)

# ==================== CÁLCULO DE RISCO E POSIÇÃO ====================

def get_balance_equity():
    """Retorna Balance e Equity da conta."""
    acc = mt5.account_info()
    if not acc: return 0.0, 0.0
    return acc.balance, acc.equity

def calculate_atr_sl_tp(symbol: str, price: float, is_buy: bool) -> Tuple[float, float, float]:
    """
    Calcula SL baseado em ATR e TP baseado em Risco/Retorno.
    Retorna: (SL Price, TP Price, ATR Value)
    """
    rates = mt5.copy_rates_from_pos(symbol, config.TIMEFRAME_MT5, 0, config.ATR_PERIOD + 20)
    
    if rates is None or len(rates) <= config.ATR_PERIOD:
        atr_value = price * 0.01 # Fallback 1%
    else:
        df = pd.DataFrame(rates)
        # Pandas TA ATR
        df.ta.atr(length=config.ATR_PERIOD, append=True)
        atr_value = df[f"ATRr_{config.ATR_PERIOD}"].iloc[-1]
        if pd.isna(atr_value) or atr_value <= 0:
            atr_value = price * 0.01

    sl_dist = atr_value * config.ATR_MULTIPLIER_SL
    tp_dist = sl_dist * config.TAKE_PROFIT_MULTIPLIER # Risco/Retorno Dinâmico

    if is_buy:
        sl = round(price - sl_dist, 2)
        tp = round(price + tp_dist, 2)
    else:
        sl = round(price + sl_dist, 2)
        tp = round(price - tp_dist, 2)
        
    return sl, tp, atr_value

def calcular_volume_inteligente(symbol):
    """
    Cálculo de Posição Institucional (Volatility Sizing).
    Risco fixo em % do Equity dividido pela distância do Stop Loss.
    """
    tick = mt5.symbol_info_tick(symbol)
    if not tick or tick.ask <= 0: return 0
    
    # 1. Dados da Conta
    balance, equity = get_balance_equity()
    if equity < 1000: return 0 # Segurança mínima
    
    # 2. Define o Risco Financeiro Máximo para este trade
    risco_financeiro = equity * (config.RISCO_POR_TRADE_PCT / 100) # Ex: 100k * 0.5% = R$ 500 de risco
    
    # 3. Calcula Distância do Stop em Reais (baseado em ATR)
    _, _, atr = calculate_atr_sl_tp(symbol, tick.ask, True)
    stop_dist_reais = atr * config.ATR_MULTIPLIER_SL
    
    if stop_dist_reais <= 0: return 0
    
    # 4. Cálculo de Lotes: (Risco $ / Stop $)
    qtd_acoes = risco_financeiro / stop_dist_reais
    
    # 5. Restrições de Alavancagem e Setor
    preco = tick.ask
    valor_nocional = qtd_acoes * preco
    
    # Trava de Setor
    setor = get_setor(symbol)
    alocacao_setor = get_sector_capital_allocation().get(setor, 0)
    limite_setor = equity * config.MAX_EXPOSURE_PER_SECTOR_PCT
    
    if (alocacao_setor + valor_nocional) > limite_setor:
        # Reduz lote para caber no setor
        espaco_livre = max(0, limite_setor - alocacao_setor)
        qtd_acoes = min(qtd_acoes, espaco_livre / preco)

    # Arredonda para lote padrão de 100
    volume = (int(qtd_acoes) // 100) * 100
    
    # Mínimo 100 se o risco permitir pelo menos 80% do lote
    if volume == 0 and qtd_acoes >= 80: volume = 100
    
    return max(0, volume)

# ==================== DADOS TÉCNICOS E MACRO ====================

def get_ibov_ticker():
    # Lógica simples para pegar o WIN atual
    # Para produção, idealmente usar a biblioteca específica ou lógica de data
    # Aqui vamos usar o configurado ou tentar deduzir
    return config.IBOV_SYMBOL_FALLBACK.replace("$N", "J26") # Exemplo simplificado

def get_market_regime():
    """
    Analisa o IBOV (ou WIN) para determinar o regime de mercado.
    Retorna: (Tendência Alta/Baixa, Preço, MM200)
    """
    symbol = "WIN$" # Símbolo contínuo no Profit/alguns brokers ou usar o ativo atual
    # Tenta achar um símbolo válido para o índice
    candidates = ["WIN$N", "WINJ26", "WINM26", "IND$N", "BOVA11"]
    ticker = None
    for c in candidates:
        if mt5.symbol_info(c) is not None:
            ticker = c
            break
            
    if not ticker: return True, 0, 0 # Fail-open (assume alta se não tiver dados)

    rates = mt5.copy_rates_from_pos(ticker, mt5.TIMEFRAME_D1, 0, config.IBOV_MA_PERIOD + 10)
    if rates is None or len(rates) < config.IBOV_MA_PERIOD:
        return True, 0, 0
        
    df = pd.DataFrame(rates)
    ma200 = df['close'].rolling(config.IBOV_MA_PERIOD).mean().iloc[-1]
    curr_price = df['close'].iloc[-1]
    
    is_bullish = curr_price > ma200
    return is_bullish, curr_price, ma200

def get_asset_technical_data(symbol):
    """Scan completo com ADX."""
    # Otimizado para Compra, mas as EMAs e RSI são calculados de forma idêntica para Venda
    p_compra = config.PARAMETROS_OTIMIZADOS_COMPRA
    
    tick = mt5.symbol_info_tick(symbol)
    if not tick: return None, "Tick Inválido"

    # Pega dados suficientes para D1 e M5
    rates_d1 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_D1, 0, 100)
    rates_m5 = mt5.copy_rates_from_pos(symbol, config.TIMEFRAME_MT5, 0, 200)
    
    if rates_d1 is None or len(rates_d1) < 50 or rates_m5 is None or len(rates_m5) < 100:
        return None, "Sem Dados Históricos"

    df_d1 = pd.DataFrame(rates_d1)
    df_m5 = pd.DataFrame(rates_m5)

    # Indicadores M5 (Pandas TA)
    # EMA: Usa os parâmetros de COMPRA apenas para definir os comprimentos
    # No bot.py, a lógica de trade vai checar qual conjunto de EMAs é melhor para C/V
    df_m5["EMA_FAST"] = ta.ema(df_m5["close"], length=p_compra["ema_fast"])
    df_m5["EMA_SLOW"] = ta.ema(df_m5["close"], length=p_compra["ema_slow"])
    # RSI
    df_m5["RSI"] = ta.rsi(df_m5["close"], length=config.RSI_PERIOD)
    # ADX
    adx_df = ta.adx(df_m5["high"], df_m5["low"], df_m5["close"], length=config.ADX_PERIOD)
    # O retorno do adx é um DF com ADX, DMP, DMN. Pegamos a coluna ADX_14
    col_adx = f"ADX_{config.ADX_PERIOD}"
    df_m5["ADX"] = adx_df[col_adx] if col_adx in adx_df else 0

    # Volume
    df_m5["VOL_MA"] = ta.sma(df_m5["tick_volume"], length=config.VOLUME_MA_PERIOD)

    # Momentum D1
    price_last = tick.last
    prev_close_d1 = float(df_d1.iloc[-2]["close"])
    momentum = ((price_last / prev_close_d1) - 1) * 100
    
    # Snapshot Última Barra Fechada (Index -2 pois -1 é a atual em formação)
    last = df_m5.iloc[-2]
    
    data = {
        "symbol": symbol, 
        "price": price_last,
        "RSI": float(last["RSI"]),
        "ADX": float(last["ADX"]),
        "EMA_FAST": float(last["EMA_FAST"]),
        "EMA_SLOW": float(last["EMA_SLOW"]),
        "momentum": float(momentum),
        "curr_vol": int(df_m5.iloc[-1]["tick_volume"]), # Volume atual (barra aberta)
        "avg_vol": float(last["VOL_MA"]),
        "spread": float(tick.ask - tick.bid),
        "tick": tick
    }
    return data, None

# ==================== AUXILIARES GERAIS ====================

def get_setor(symbol):
    base = symbol.replace(config.CORRETORA_SUFFIX, "")
    return config.SETORES.get(base, "OUTROS")

def get_sector_capital_allocation():
    alocacao = defaultdict(float)
    positions = mt5.positions_get() or []
    for p in positions:
        valor_financeiro = p.volume * p.price_current
        alocacao[get_setor(p.symbol)] += valor_financeiro
    return alocacao

def get_daily_profit_loss():
    # Histórico de hoje
    now = datetime.now()
    inicio_dia = now.replace(hour=0, minute=0, second=0, microsecond=0)
    
    deals = mt5.history_deals_get(inicio_dia, now)
    realized_pl = sum(d.profit for d in deals) if deals else 0.0
    
    positions = mt5.positions_get()
    floating_pl = sum(p.profit for p in positions) if positions else 0.0
    
    total_pl = realized_pl + floating_pl
    
    acc = mt5.account_info()
    balance = acc.balance if acc else 1.0
    
    return total_pl, (total_pl / balance) * 100

def is_trading_time():
    now = datetime.now().time()
    return config.START_TIME <= now <= config.END_TIME

def register_trade(symbol, side, price, volume, result, sl, tp):
    os.makedirs("trades", exist_ok=True)
    with open(config.CSV_FILE, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
            symbol, side, f"{price:.2f}", volume, result, f"SL:{sl}", f"TP:{tp}"
        ])

def close_all_positions():
    logger.warning("FECHAMENTO DE EMERGÊNCIA ACIONADO.")
    positions = mt5.positions_get()
    for pos in positions:
        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": pos.symbol,
            "volume": pos.volume,
            "type": mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
            "position": pos.ticket,
            "magic": pos.magic,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        mt5.order_send(req)

# Trailing Stop Thread (Simplificada)
trailing_running = False
def trailing_stop_thread():
    global trailing_running
    trailing_running = True
    logger.info("Serviço de Trailing Stop Ativo.")
    
    while trailing_running:
        try:
            for pos in mt5.positions_get() or []:
                if pos.magic != 20251230: continue
                
                # Recalcula ATR
                _, _, atr = calculate_atr_sl_tp(pos.symbol, pos.price_current, True)
                
                # Lógica Trailing: Move SL se lucro > 1x ATR
                delta_price = pos.price_current - pos.price_open if pos.type == 0 else pos.price_open - pos.price_current
                
                if delta_price > (atr * 1.0): # Se lucro > 1 ATR
                    new_sl = pos.price_current - (atr * 1.5) if pos.type == 0 else pos.price_current + (atr * 1.5)
                    
                    # Verifica se o novo SL é melhor que o atual
                    update = False
                    if pos.type == 0 and (pos.sl == 0 or new_sl > pos.sl): update = True
                    if pos.type == 1 and (pos.sl == 0 or new_sl < pos.sl): update = True
                    
                    if update:
                        req = {"action": mt5.TRADE_ACTION_SLTP, "position": pos.ticket, "sl": new_sl, "tp": pos.tp}
                        mt5.order_send(req)
                        
        except Exception as e:
            logger.error(f"Erro no Trailing: {e}")
        time.sleep(0.3)