import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import math
import logging
import os
import json
from typing import Dict, Any, List

logger = logging.getLogger("utils")

# ============================================
#  SAFE COPY RATES
# ============================================
def safe_copy_rates(symbol: str, timeframe, count: int = 200):
    """
    Wrapper mais seguro para copy_rates_from_pos
    """
    try:
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
        if rates is None or len(rates) == 0:
            logger.warning(f"safe_copy_rates: sem dados para {symbol}")
            return None

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df = df.set_index("time").sort_index()
        return df

    except Exception as e:
        logger.exception(f"safe_copy_rates ERROR {symbol}: {e}")
        return None


# ============================================
#  ATR
# ============================================
def get_atr(df: pd.DataFrame, period: int = 14):
    """
    ATR clássico de Wilder
    """
    try:
        high = df["high"]
        low = df["low"]
        close = df["close"]

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        return float(atr.iloc[-1])

    except Exception as e:
        logger.exception(f"get_atr ERROR: {e}")
        return None


# ============================================
#  CÁLCULO DE TAMANHO DE POSIÇÃO POR RISCO
# ============================================
def calculate_position_size(symbol: str, sl_price: float, risk_pct: float = 0.01):
    """
    Calcula o volume baseado no risco (1% por padrão)
    Lógica: risk_money = equity * risk_pct
    stop_distance = abs(entry - SL)
    volume = risk_money / (stop_distance * contract_size)
    """
    try:
        acc_info = mt5.account_info()
        if acc_info is None:
            logger.warning("calculate_position_size: sem account_info()")
            return None

        equity = acc_info.equity
        risk_money = equity * risk_pct

        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logger.warning(f"calculate_position_size: symbol_info None for {symbol}")
            return None

        tick = mt5.symbol_info_tick(symbol)
        entry = tick.ask if tick and tick.ask > 0 else tick.bid

        stop_distance = abs(entry - sl_price)
        if stop_distance <= 0:
            logger.warning("calculate_position_size: stop distance invalid")
            return None

        contract_size = symbol_info.trade_contract_size or 1
        volume = risk_money / (stop_distance * contract_size)

        # Ajustar para step de volume permitido
        vol_step = symbol_info.volume_step
        volume = max(symbol_info.volume_min, round(volume / vol_step) * vol_step)

        return float(volume)

    except Exception as e:
        logger.exception(f"calculate_position_size ERROR: {e}")
        return None


# ============================================
#  EQUITY
# ============================================
def get_account_equity():
    try:
        acc = mt5.account_info()
        if acc:
            return float(acc.equity)
        return 0.0
    except:
        return 0.0

# ============================================
#  ENVIO DE ORDEM (BUY/SELL) COM SL/TP
# ============================================
def send_order_with_sl_tp(symbol: str, side: str, volume: float, sl: float, tp: float):
    try:
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            return {"success": False, "reason": "symbol_info None"}

        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            return {"success": False, "reason": "no tick"}

        # Preço atual
        price = tick.ask if side.upper() == "BUY" else tick.bid

        tick_size = symbol_info.trade_tick_size or 0.01

        # Arredonda SL/TP para tick correto
        sl = round(sl / tick_size) * tick_size
        tp = round(tp / tick_size) * tick_size

        # Garantir distância mínima do preço (importante na B3)
        # Assumindo que 2 ticks é o mínimo.
        if side.upper() == "BUY":
            if sl >= price:
                sl = price - (2 * tick_size)
            if tp <= price:
                tp = price + (2 * tick_size)
        else:  # SELL
            if sl <= price:
                sl = price + (2 * tick_size)
            if tp >= price:
                tp = price - (2 * tick_size)

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": mt5.ORDER_TYPE_BUY if side.upper() == "BUY" else mt5.ORDER_TYPE_SELL,
            "price": price,
            "sl": sl,
            "tp": tp,
            "magic": 99,
            "comment": "bot_fast",
            "type_filling": mt5.ORDER_FILLING_IOC,  
        }

        result = mt5.order_send(request)

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return {"success": False, "reason": result.comment}

        return {"success": True, "order": result.order}

    except Exception as e:
        logger.exception(f"send_order_with_sl_tp ERROR: {e}")
        return {"success": False, "reason": str(e)}

# =========================================================
# FUNÇÕES DE CACHE DO OPTIMIZER (movidas para utils.py)
# =========================================================
def load_optimized_summaries(symbols: List[str], base_dir: str) -> Dict[str, Dict[str, Any]]:
    """
    Carrega o arquivo de resumo JSON (gerado pelo optimizer) para cada símbolo.
    Retorna um dicionário {symbol: best_params}.
    """
    logger.info("Tentando carregar summaries otimizados...")
    optimized_params: Dict[str, Dict[str, Any]] = {}
    
    # O diretório de saída do optimizer
    output_dir = os.path.join(base_dir, "optimizer_output")
    
    for symbol in symbols:
        file_path = None
        
        # 1. Tenta carregar o formato novo: {symbol}_summary.json
        new_format_path = os.path.join(output_dir, f"{symbol}_summary.json")
        if os.path.exists(new_format_path):
            file_path = new_format_path
        
        # 2. Tenta carregar o formato antigo: summary_{regime}_{symbol}_*.json (pega o mais recente)
        if file_path is None:
            try:
                files = [f for f in os.listdir(output_dir) if f.startswith("summary_") and f"_{symbol}_" in f and f.endswith(".json")]
                if files:
                    files.sort(reverse=True) # newest first
                    file_path = os.path.join(output_dir, files[0])
            except Exception:
                 continue
            
        if file_path is None:
            logger.debug(f"Cache: Arquivo de resumo não encontrado para {symbol}.")
            continue
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            params = None
            # Tenta extrair do formato 'COMBO' (novo)
            best_combo = data.get("COMBO")
            if best_combo:
                params = best_combo.get("best_params")
                
            # Tenta extrair do formato 'top_params' (antigo)
            elif data.get("top_params"):
                params = data["top_params"][0]

            if params:
                # Mapeia as chaves do optimizer para o que o bot espera (usando defaults se faltar)
                optimized_params[symbol] = {
                    "ema_short": int(params.get("ema_short", 9)),
                    "ema_long": int(params.get("ema_long", 21)),
                    "rsi_period": int(params.get("rsi_period", 14)),
                    "rsi_low": float(params.get("rsi_oversold", params.get("rsi_low", 30.0))), 
                    "rsi_high": float(params.get("rsi_overbought", params.get("rsi_high", 70.0))),
                    "mom_min": float(params.get("mom_threshold", params.get("mom_min", 0.0))),
                    "adx_period": int(params.get("adx_period", 14)),
                }
            else:
                logger.warning(f"Cache: Parâmetros válidos ausentes no resumo de {symbol}.")

        except json.JSONDecodeError:
            logger.error(f"Cache: Erro de JSON no arquivo de {symbol}: {file_path}")
        except Exception as e:
            logger.error(f"Cache: Erro inesperado ao carregar {symbol}: {e}")

    logger.info(f"Cache: Carregados {len(optimized_params)} resumos otimizados com sucesso.")
    return optimized_params

def get_optimized_params(symbol: str, optimized_cache: Dict[str, Dict[str, Any]], default_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Retorna os parâmetros otimizados para um símbolo ou os padrões (DEFAULT) se não encontrados.
    """
    params = optimized_cache.get(symbol)
    
    # Verifica se os parâmetros chave (EMA) estão presentes e são válidos
    if params and all(v != 0 and v is not None for k, v in params.items() if k in ["ema_short", "ema_long"]):
        return params
    
    # Se otimização falhou ou não existe, retorna o default
    return default_params