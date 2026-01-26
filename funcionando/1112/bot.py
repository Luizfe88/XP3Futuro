#!/usr/bin/env python3
"""
bot_final_corrigido.py
Bot FAST+SLOW com:
 - cache de summaries do optimizer (carregado no slow loop)
 - painel colorido + tabela alinhada
 - leitura segura de volumes (tick_volume/volume)
 - integração com utils (safe_copy_rates, get_atr, calculate_position_size, send_order_with_sl_tp, get_account_equity)
 - proteção contra warnings repetidos
 - operação em M5 (configurable)
"""

import os
import time
import threading
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional
import MetaTrader5 as mt5
import pandas as pd
import numpy as np

# Try colorama for Windows ANSI support
try:
    from colorama import init as colorama_init, Fore, Style
    colorama_init(autoreset=True)
    C_GREEN = Fore.GREEN; C_RED = Fore.RED; C_YELLOW = Fore.YELLOW; C_CYAN = Fore.CYAN; C_MAGENTA = Fore.MAGENTA; C_RESET = Style.RESET_ALL; C_BOLD = Style.BRIGHT
except Exception:
    C_GREEN = "\033[92m"; C_RED = "\033[91m"; C_YELLOW = "\033[93m"; C_CYAN = "\033[96m"; C_MAGENTA = "\033[95m"; C_RESET = "\033[0m"; C_BOLD = "\033[1m"

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("bot_final")

# ---------------------------
# Load project config (tries both)
# ---------------------------
try:
    import config
except Exception:
    try:
        import config_mod as config
    except Exception:
        config = None

# If config missing, set defaults
DEFAULT_OPTIMIZER_OUTPUT = r"C:\Users\luizf\Documents\xp3v2\optimizer_output"
OPTIMIZER_OUTPUT = getattr(config, "OPTIMIZER_OUTPUT", DEFAULT_OPTIMIZER_OUTPUT) if config else DEFAULT_OPTIMIZER_OUTPUT

# ---------------------------
# Utils import (prefer project utils)
# ---------------------------
utils = None
for name in ("utils", "utils_corrigido", "utils_mod", "utils_final"):
    try:
        utils = __import__(name)
        break
    except Exception:
        utils = None

# ---------------------------
# Parameters (can override in config)
# ---------------------------
FAST_INTERVAL_SEC = 0.03  # 30ms
SLOW_INTERVAL_SEC = getattr(config, "SCAN_INTERVAL_SECONDS", 60) if config else 60
TF_STR = getattr(config, "TIMEFRAME_DEFAULT", "M5") if config else "M5"
TF_MAP = {"M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15, "H1": mt5.TIMEFRAME_H1}
TIMEFRAME_MT5 = TF_MAP.get(TF_STR, mt5.TIMEFRAME_M5)
EMA_FAST_N = getattr(config, "DEFAULT_PARAMS", {}).get("ema_short", 9)
EMA_SLOW_N = getattr(config, "DEFAULT_PARAMS", {}).get("ema_long", 21)
RISK_PCT = 0.01
SL_ATR_MULT = 2.0
TP_ATR_MULT = 3.0
MAX_SYMBOLS = getattr(config, "MAX_SYMBOLS", 10) if config else 10

# ---------------------------
# state + caches
# ---------------------------
_symbol_state: Dict[str, Dict[str, Any]] = {}
_recent_orders: Dict[str, float] = {}
_order_cooldown = getattr(config, "DUP_ORDER_COOLDOWN_SECONDS", 60) if config else 60

# optimizer caches (loaded/updated from slow loop)
optimizer_cache: Dict[str, Dict[str, Any]] = {}
missing_optimizer_warned = set()

# ---------------------------
# helpers
# ---------------------------
def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")

def fmt(x, prec=4):
    if x is None:
        return "N/A"
    try:
        return f"{float(x):.{prec}f}"
    except Exception:
        return str(x)

def order_signature(symbol: str, side: str, volume: float, sl: float, tp: Optional[float]) -> str:
    return f"{symbol}|{side}|{round(volume,6)}|{round(sl,6)}|{round(tp or 0.0,6)}"

def is_duplicate_recent(sig: str) -> bool:
    now = time.time()
    # cleanup to keep dict small
    to_del = [k for k,v in _recent_orders.items() if now - v > (_order_cooldown*5)]
    for k in to_del:
        _recent_orders.pop(k, None)
    last = _recent_orders.get(sig)
    if last and (now - last) < _order_cooldown:
        return True
    return False

def record_order(sig: str):
    _recent_orders[sig] = time.time()

# ---------------------------
# Load optimizer top_params (Modo A) with cache - AGORA USA O CACHE CENTRALIZADO
# ---------------------------
def load_optimizer_targets(symbol: str) -> dict:
    """
    Retorna os parâmetros otimizados para um símbolo (ou defaults se cache falhar).
    Mapeia a estrutura de saída do optimizer (ema_short/long, rsi_low/high) 
    para as chaves internas do bot (ema_fast/slow, rsi_low/high, mom_min, adx_period).
    """
    # 1. Definir os defaults baseados em config.py
    config_defaults = {
        "ema_short": EMA_FAST_N,
        "ema_long": EMA_SLOW_N,
        "rsi_period": 14,
        "rsi_low": 40.0, 
        "rsi_high": 60.0,
    }
    # Tenta carregar config se estiver disponível, para usar os defaults corretos
    if config:
        config_defaults["ema_short"] = config.DEFAULT_PARAMS.get("ema_short", EMA_FAST_N)
        config_defaults["ema_long"] = config.DEFAULT_PARAMS.get("ema_long", EMA_SLOW_N)
        config_defaults["rsi_low"] = config.DEFAULT_PARAMS.get("rsi_oversold", 40.0)
        config_defaults["rsi_high"] = config.DEFAULT_PARAMS.get("rsi_overbought", 60.0)

    # 2. Tentar carregar a partir do cache usando a função utilitária
    optimized_data = config_defaults
    
    if utils and hasattr(utils, "get_optimized_params"):
        # Popula optimized_data com os valores do cache, se disponíveis
        optimized_data = utils.get_optimized_params(symbol, optimizer_cache, config_defaults)

    # 3. Mapear para o formato interno que o resto do bot espera (ema_fast/slow)
    return {
        # EMA
        "ema_fast": optimized_data.get("ema_short", config_defaults["ema_short"]),
        "ema_slow": optimized_data.get("ema_long", config_defaults["ema_long"]),
        # RSI
        "rsi_low": optimized_data.get("rsi_low", config_defaults["rsi_low"]),
        "rsi_high": optimized_data.get("rsi_high", config_defaults["rsi_high"]),
        # MOM & ADX (Pegando diretamente do resultado otimizado ou default)
        "mom_min": optimized_data.get("mom_min", config.DEFAULT_PARAMS.get("mom_threshold", 0.0) if config else 0.0),
        "adx_period": optimized_data.get("adx_period", config.DEFAULT_PARAMS.get("adx_period", 14) if config else 14),
    }

# ---------------------------
# init symbol state (from history)
# ---------------------------
def init_symbol_state(symbol: str):
    try:
        if utils and hasattr(utils, "safe_copy_rates"):
            df = utils.safe_copy_rates(symbol, TIMEFRAME_MT5, 500)
        else:
            raw = mt5.copy_rates_from_pos(symbol, TIMEFRAME_MT5, 0, 500)
            df = pd.DataFrame(raw) if raw is not None and len(raw)>0 else None
            if df is not None:
                df.columns = [c.lower() for c in df.columns]
                if "time" in df.columns:
                    df["time"] = pd.to_datetime(df["time"], unit="s")
                    df = df.set_index("time").sort_index()

        if df is None or df.empty:
            logger.warning(f"init_symbol_state: sem dados para {symbol}")
            return None

        close = df['close'].astype(float)
        ema_fast = float(close.ewm(span=EMA_FAST_N, adjust=False).mean().iloc[-1])
        ema_slow = float(close.ewm(span=EMA_SLOW_N, adjust=False).mean().iloc[-1])

        vol_buffer = []
        if "volume" in df.columns and df['volume'].notna().any():
            recent_vols = df['volume'].dropna().astype(float).tail(20).tolist()
            vol_buffer = recent_vols.copy()

        _symbol_state[symbol] = {
            "ema_fast": ema_fast,
            "ema_slow": ema_slow,
            "last_price": float(close.iloc[-1]),
            "last_time": df.index[-1].to_pydatetime(),
            "targets": load_optimizer_targets(symbol), # USA A NOVA LÓGICA DE CARREGAMENTO
            "vol_buffer": vol_buffer,
            "last_vol": float(df['volume'].iloc[-1]) if "volume" in df.columns and len(df['volume'])>0 else 0.0,
            "vol_mean": float(np.mean(vol_buffer)) if vol_buffer else 0.0
        }
        logger.info(f"Initialized state for {symbol}: EMAf={ema_fast:.4f}, EMAs={ema_slow:.4f}")
        return _symbol_state[symbol]
    except Exception as e:
        logger.exception(f"init_symbol_state failed for {symbol}: {e}")
        return None

# ---------------------------
# update EMA incremental
# ---------------------------
def update_ema_incremental(symbol: str, price: float):
    s = _symbol_state.get(symbol)
    if not s:
        return
    af = 2.0 / (EMA_FAST_N + 1.0)
    aslow = 2.0 / (EMA_SLOW_N + 1.0)
    s["ema_fast"] = af * price + (1 - af) * s.get("ema_fast", price)
    s["ema_slow"] = aslow * price + (1 - aslow) * s.get("ema_slow", price)
    s["last_price"] = price
    s["last_time"] = datetime.now(timezone.utc)

# ---------------------------
# indicators snapshot (light) used in slow loop when needed
# ---------------------------
def indicators_snapshot_with_diagnostics(symbol: str, timeframe, lookback=300):
    """
    Retorna dicionário com indicadores calculados a partir das candles (single fetch).
    Usado pelo slow loop para montar painel quando não há estado incremental completo.
    """
    result = {"symbol": symbol}
    try:
        if utils and hasattr(utils, "safe_copy_rates"):
            df = utils.safe_copy_rates(symbol, timeframe, lookback)
        else:
            raw = mt5.copy_rates_from_pos(symbol, timeframe, 0, lookback)
            df = pd.DataFrame(raw) if raw is not None and len(raw)>0 else None
            if df is not None:
                df.columns = [c.lower() for c in df.columns]
                if "time" in df.columns:
                    df["time"] = pd.to_datetime(df["time"], unit="s")
                    df = df.set_index("time").sort_index()

        if df is None or df.empty:
            result["error"] = "no_data"
            return result

        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)

        ema_fast = float(close.ewm(span=EMA_FAST_N, adjust=False).mean().iloc[-1])
        ema_slow = float(close.ewm(span=EMA_SLOW_N, adjust=False).mean().iloc[-1])

        # RSI (simple)
        delta = close.diff()
        up = delta.clip(lower=0).rolling(14).mean()
        down = -delta.clip(upper=0).rolling(14).mean()
        rs = up / down
        rsi = float((100 - (100 / (1 + rs))).iloc[-1]) if not rs.empty else None

        mom = float(close.iloc[-1] / close.shift(10).iloc[-1] - 1) if len(close)>10 else 0.0

        # ATR via utils or fallback
        atr = None
        try:
            if utils and hasattr(utils, "get_atr"):
                atr = utils.get_atr(df, 14)
            else:
                tr1 = (high - low).abs()
                tr2 = (high - close.shift(1)).abs()
                tr3 = (low - close.shift(1)).abs()
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                atr = float(tr.rolling(14).mean().iloc[-1])
        except Exception:
            atr = None

        # ADX fallback N/A (calculation heavy); try to use pandas_ta if available via utils
        adx = None
        # O ADX é omitido aqui para manter a leveza, a não ser que tenha o pandas_ta

        # volume
        if "tick_volume" in df.columns:
            last_vol = float(df["tick_volume"].iloc[-1])
            vol_mean = float(df["tick_volume"].tail(20).mean())
        elif "volume" in df.columns:
            last_vol = float(df["volume"].iloc[-1])
            vol_mean = float(df["volume"].tail(20).mean())
        else:
            last_vol = 0.0
            vol_mean = 0.0

        result.update({
            "ema_fast": ema_fast,
            "ema_slow": ema_slow,
            "rsi": rsi,
            "mom": mom,
            "atr": atr,
            "adx": adx,
            "last_vol": last_vol,
            "vol_mean": vol_mean,
            "status": "OK"
        })
        return result

    except Exception as e:
        logger.exception(f"indicators_snapshot failed {symbol}: {e}")
        result["error"] = "exception"
        return result

# ---------------------------
# render panel (color + aligned table)
# ---------------------------
def render_panel(equity: float, portfolio_indicators: Dict[str, Dict[str,Any]], positions: List[Dict[str,Any]]):
    clear_screen()
    now = datetime.now(timezone.utc) - timedelta(hours=3)
    now_str = now.strftime("%Y-%m-%d %H:%M:%S UTC-3")
    print(C_BOLD + C_MAGENTA + "===== FAST BOT PANEL =====" + C_RESET)
    print(f"{C_YELLOW}Relógio: {now_str}{C_RESET}    Fast loop: {FAST_INTERVAL_SEC*1000:.0f}ms    Slow loop: {SLOW_INTERVAL_SEC}s")
    if equity is None or equity == 0:
        print(C_RED + "AVISO: Equity retornou 0.00 — verifique MT5." + C_RESET)
    print(f"Equity: {C_BOLD}{C_YELLOW}{equity:,.2f}{C_RESET}\n")

    # Positions
    print(C_BOLD + C_CYAN + "=== POSIÇÕES EM CARTEIRA ===" + C_RESET)
    if not positions:
        print("  (nenhuma posição aberta)")
    else:
        for p in positions:
            prof = p.get("profit", 0.0)
            prof_col = C_GREEN if prof >= 0 else C_RED
            vol = p.get("volume", 0.0)
            price = p.get("price", 0.0)
            print(f"  {p['symbol']:6s} Vol:{vol:10.2f} Price:{price:12.4f} Profit:{prof_col}{prof:10.2f}{C_RESET}")
    print("")

    # Header for table
    hdr = f"{'SYM':6s} {'EMA (f>s)':20s} {'RSI':6s} {'MOM':9s} {'ATR':8s} {'ADX':6s} {'VOL (L/M)':12s} {'STATUS':8s} {'OPT_TARGETS'}"
    print(C_BOLD + C_MAGENTA + hdr + C_RESET)

    for sym, d in portfolio_indicators.items():
        if not d or d.get("error"):
            err = d.get("error") if isinstance(d, dict) else "no_data"
            print(f"{sym:6s} {C_RED}ERROR:{err}{C_RESET}")
            continue

        # Use dynamic targets if present
        targets = d.get("targets", load_optimizer_targets(sym))

        ema_f = d.get("ema_fast")
        ema_s = d.get("ema_slow")
        ema_ok = (ema_f is not None and ema_s is not None and ema_f > ema_s)

        rsi = d.get("rsi")
        rsi_ok = (rsi is not None and targets.get("rsi_low",40) <= rsi <= targets.get("rsi_high",60))

        mom = d.get("mom")
        mom_ok = (mom is not None and mom >= targets.get("mom_min", 0.0))

        last_vol = d.get("last_vol", 0.0) or 0.0
        vol_mean = d.get("vol_mean", 0.0) or 0.0
        vol_ok = (last_vol >= vol_mean and vol_mean > 0)

        adx = d.get("adx")
        adx_ok = (adx is None) or (adx >= targets.get("adx_period", 14))

        status_ok = ema_ok and rsi_ok and mom_ok and vol_ok and adx_ok

        ema_str = f"{(ema_f or 0):.2f}>{(ema_s or 0):.2f}"
        rsi_str = f"{rsi:.1f}" if rsi is not None else "N/A"
        mom_str = f"{mom:.4f}" if mom is not None else "N/A"
        atr_str = f"{d.get('atr'):.4f}" if d.get('atr') is not None else "N/A"
        adx_str = f"{adx:.2f}" if adx is not None else "N/A"
        vol_str = f"{int(last_vol)}/{int(vol_mean)}" if vol_mean>0 else f"{int(last_vol)}/0"

        status_col = C_GREEN if status_ok else C_RED
        ema_col = C_GREEN if ema_ok else C_RED
        rsi_col = C_GREEN if rsi_ok else C_RED
        mom_col = C_GREEN if mom_ok else C_RED
        vol_col = C_GREEN if vol_ok else C_RED
        adx_col = C_GREEN if adx_ok else C_RED

        tgt_str = f"EMA {targets.get('ema_fast')}/{targets.get('ema_slow')} RSI {targets.get('rsi_low')}-{targets.get('rsi_high')} MOM>{targets.get('mom_min')}"

        print(f"{sym:6s} {ema_col}{ema_str:20s}{C_RESET} {rsi_col}{rsi_str:6s}{C_RESET} {mom_col}{mom_str:9s}{C_RESET} {atr_str:8s} {adx_col}{adx_str:6s}{C_RESET} {vol_col}{vol_str:12s}{C_RESET} {status_col}{('OK' if status_ok else 'NO'):8s}{C_RESET} {C_CYAN}{tgt_str}{C_RESET}")

    print("\n" + C_BOLD + C_MAGENTA + "==========================" + C_RESET)

# ---------------------------
# fast loop: incremental, minimal IO
# ---------------------------
def fast_loop(symbols: List[str]):
    logger.info(f"Fast loop started. Interval: {FAST_INTERVAL_SEC}s")
    # ensure initial states
    for s in symbols:
        init_symbol_state(s)
    while True:
        try:
            for sym in symbols:
                try:
                    tick = mt5.symbol_info_tick(sym)
                    if tick is None:
                        continue
                    price = float(tick.ask) if getattr(tick, "ask", None) and tick.ask>0 else float(tick.bid)
                    if sym not in _symbol_state:
                        init_symbol_state(sym)
                        if sym not in _symbol_state:
                            continue
                    update_ema_incremental(sym, price)
                    state = _symbol_state[sym]

                    # light checks: compute real indicators only when needed (e.g., ema cross)
                    targets = state.get("targets", load_optimizer_targets(sym))
                    ema_fast = state.get("ema_fast")
                    ema_slow = state.get("ema_slow")

                    # determine trade side from targets (default COMPRA)
                    side = "BUY" if targets.get("side","COMPRA") != "VENDA" else "SELL"
                    ema_cross = (ema_fast > ema_slow) if side=="BUY" else (ema_fast < ema_slow)

                    do_full_check = False
                    if ema_cross:
                        do_full_check = True

                    if do_full_check:
                        # get small window to compute RSI/ATR/MOM/volume/adx (but this is done infrequently since cross rarely flips)
                        if utils and hasattr(utils, "safe_copy_rates"):
                            df = utils.safe_copy_rates(sym, TIMEFRAME_MT5, 300)
                        else:
                            raw = mt5.copy_rates_from_pos(sym, TIMEFRAME_MT5, 0, 300)
                            df = pd.DataFrame(raw) if raw is not None and len(raw)>0 else None
                            if df is not None:
                                df.columns = [c.lower() for c in df.columns]
                                if "time" in df.columns:
                                    df["time"] = pd.to_datetime(df["time"], unit="s")
                                    df = df.set_index("time").sort_index()

                        if df is None or df.empty:
                            continue

                        # RSI
                        delta = df['close'].astype(float).diff()
                        up = delta.clip(lower=0).rolling(14).mean()
                        down = -delta.clip(upper=0).rolling(14).mean()
                        rs = up / down
                        rsi = float((100 - (100 / (1 + rs))).iloc[-1]) if not rs.empty else None

                        # MOM
                        # Nota: targets.get("mom_min", 10) aqui é usado como período, o que pode ser um erro de design
                        # Assumindo que o período MOM é 10, e targets.get("mom_min") é o threshold.
                        mom = float(df['close'].astype(float).iloc[-1] / df['close'].astype(float).shift(10).iloc[-1] - 1) if len(df)>10 else 0.0

                        # ATR
                        atr = None
                        try:
                            if utils and hasattr(utils, "get_atr"):
                                atr = utils.get_atr(df, 14)
                        except Exception:
                            atr = None

                        # volume
                        if 'tick_volume' in df.columns:
                            last_vol = float(df['tick_volume'].iloc[-1])
                            vol_mean = float(df['tick_volume'].tail(20).mean())
                        elif 'volume' in df.columns:
                            last_vol = float(df['volume'].iloc[-1])
                            vol_mean = float(df['volume'].tail(20).mean())
                        else:
                            last_vol = 0.0
                            vol_mean = 0.0

                        # ADX via utils/pandas_ta optional (fast loop keep light)
                        adx = None

                        # update snapshot state
                        s = _symbol_state.get(sym, {})
                        vb = s.get("vol_buffer", [])
                        vb.append(last_vol if last_vol is not None else 0.0)
                        if len(vb) > 20: vb.pop(0)
                        s["vol_buffer"] = vb
                        s["last_vol"] = last_vol if last_vol is not None else 0.0
                        s["vol_mean"] = float(np.mean(vb)) if vb else 0.0
                        s["rsi"] = rsi
                        s["mom"] = mom
                        s["atr"] = atr
                        s["adx"] = adx
                        s["targets"] = targets
                        _symbol_state[sym] = s

                        # check trade conditions
                        rsi_ok = (rsi is not None) and (targets.get("rsi_low",40) <= rsi <= targets.get("rsi_high",60))
                        mom_ok = (mom is not None) and (mom >= targets.get("mom_min", 0.0))
                        vol_ok = (s.get("last_vol",0) >= s.get("vol_mean",0) and s.get("vol_mean",0)>0)
                        adx_ok = (adx is None) or (adx >= targets.get("adx_period", 14))
                        trade_ok = rsi_ok and mom_ok and vol_ok and adx_ok and ema_cross

                        # ensure no position exists
                        positions = mt5.positions_get(symbol=sym)
                        has_pos = bool(positions and len(positions)>0)
                        if trade_ok and not has_pos:
                            if atr is None or atr <= 0:
                                continue
                            last_price = price
                            if side=="BUY":
                                sl_price = max(0.0001, last_price - SL_ATR_MULT * atr)
                                tp_price = last_price + TP_ATR_MULT * atr
                            else:
                                sl_price = last_price + SL_ATR_MULT * atr
                                tp_price = last_price - TP_ATR_MULT * atr

                            lots = None
                            if utils and hasattr(utils, "calculate_position_size"):
                                lots = utils.calculate_position_size(sym, sl_price, risk_pct=RISK_PCT)
                            if not lots or lots <= 0:
                                logger.debug(f"{sym}: lots inválido {lots}")
                            else:
                                sig = order_signature(sym, side, lots, sl_price, tp_price)
                                if is_duplicate_recent(sig):
                                    logger.debug(f"{sym}: duplicate recent, skip order")
                                else:
                                    if utils and hasattr(utils, "send_order_with_sl_tp"):
                                        res = utils.send_order_with_sl_tp(sym, side, lots, sl_price, tp_price)
                                    else:
                                        res = {"success": False, "reason": "no_send_func"}
                                    if isinstance(res, dict) and res.get("success"):
                                        logger.info(f"[order] EXECUTED {sym} {side} lots={lots} sl={sl_price:.4f} tp={tp_price:.4f}")
                                        record_order(sig)
                                    else:
                                        logger.warning(f"[order] failed for {sym}: {res}")
                    # end do_full_check
                except Exception as inner:
                    logger.exception(f"fast_loop inner exception for {sym}: {inner}")
            # tiny sleep between symbols
            time.sleep(0.0005)
        except Exception as e:
            logger.exception(f"fast_loop top exception: {e}")
        # wait full fast interval
        time.sleep(FAST_INTERVAL_SEC)

# ---------------------------
# slow loop: builds panel and refreshes optimizer cache periodically
# ---------------------------
def slow_loop(symbols: List[str]):
    # CORREÇÃO PARA O SyntaxError: nome 'optimizer_cache' é usado antes da declaração global
    global optimizer_cache 
    
    logger.info(f"Slow loop started. Interval: {SLOW_INTERVAL_SEC}s")
    while True:
        try:
            # --- NOVO: Carregamento do cache otimizado usando utils ---
            if utils and hasattr(utils, "load_optimized_summaries"):
                logger.info(f"Slow loop: Carregando summaries otimizados de {OPTIMIZER_OUTPUT}...")
                
                # Obtém o diretório base (o bot.py está em xp3v2, o output está em xp3v2/optimizer_output)
                # O utils.load_optimized_summaries precisa do diretório pai que contém 'optimizer_output'.
                base_dir = os.path.dirname(OPTIMIZER_OUTPUT) 
                
                try:
                    # Carrega e REATRIBUI o cache completo
                    optimizer_cache = utils.load_optimized_summaries(symbols, base_dir)
                    logger.info(f"Cache do optimizer carregado. {len(optimizer_cache)} símbolos com dados.")
                    missing_optimizer_warned.clear() # Limpa warnings
                except Exception as e:
                    logger.exception(f"Slow loop: Falha ao carregar summaries otimizados: {e}")
            else:
                logger.warning("Slow loop: utils.load_optimized_summaries não encontrado. Usando defaults.")
            # --- FIM NOVO ---
            
            # build indicators snapshot for panel
            portfolio_indicators = {}
            for sym in symbols:
                st = _symbol_state.get(sym)
                if st and ("rsi" in st):
                    merged = {
                        "symbol": sym,
                        "ema_fast": st.get("ema_fast"),
                        "ema_slow": st.get("ema_slow"),
                        "rsi": st.get("rsi"),
                        "mom": st.get("mom"),
                        "atr": st.get("atr"),
                        "adx": st.get("adx"),
                        "vol_mean": st.get("vol_mean"),
                        "last_vol": st.get("last_vol"),
                        "targets": st.get("targets", load_optimizer_targets(sym))
                    }
                    portfolio_indicators[sym] = merged
                else:
                    portfolio_indicators[sym] = indicators_snapshot_with_diagnostics(sym, TIMEFRAME_MT5, lookback=300)
                    if "targets" not in portfolio_indicators[sym]:
                        portfolio_indicators[sym]["targets"] = load_optimizer_targets(sym)

            # account + positions
            equity = utils.get_account_equity() if utils and hasattr(utils, "get_account_equity") else 0.0
            positions = []
            try:
                pos = mt5.positions_get()
                if pos:
                    for p in pos:
                        positions.append({
                            "symbol": p.symbol,
                            "volume": float(p.volume),
                            "price": float(getattr(p, "price_open", getattr(p, "price", 0.0))),
                            "profit": float(p.profit)
                        })
            except Exception:
                positions = []

            # render UI
            render_panel(equity, portfolio_indicators, positions)

            # save snapshot JSON
            try:
                os.makedirs(OPTIMIZER_OUTPUT, exist_ok=True)
                ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                snapshot_file = os.path.join(OPTIMIZER_OUTPUT, f"fast_panel_snapshot_{ts}.json")
                with open(snapshot_file, "w", encoding="utf-8") as f:
                    json.dump({"ts": ts, "equity": equity, "positions": positions, "indicators": portfolio_indicators}, f, default=str, indent=2, ensure_ascii=False)
            except Exception:
                logger.exception("Failed saving snapshot JSON (slow_loop)")

        except Exception as e:
            logger.exception(f"slow_loop exception: {e}")

        time.sleep(SLOW_INTERVAL_SEC)

# ---------------------------
# symbol selection respecting sectors
# ---------------------------
def select_symbols_with_sector_limits(symbols, max_symbols=None, max_per_sector=None):
    max_symbols = max_symbols or MAX_SYMBOLS
    max_per_sector = max_per_sector or getattr(config, "MAX_PER_SECTOR", 2) if config else 2
    selected = []
    sector_count = {}
    for sym in symbols:
        sector = None
        try:
            sector = config.SECTOR_MAP.get(sym)
        except Exception:
            sector = None
        if sector is None:
            logger.warning(f"{sym} ignorado — não possui setor definido no SECTOR_MAP.")
            continue
        if sector_count.get(sector, 0) >= max_per_sector:
            logger.info(f"{sym} ignorada → setor {sector} já atingiu limite ({max_per_sector}).")
            continue
        selected.append(sym)
        sector_count[sector] = sector_count.get(sector, 0) + 1
        if len(selected) >= max_symbols:
            break
    logger.info(f"Ações selecionadas após filtro setorial: {selected}")
    return selected

def ensure_symbol_visible(symbol):
    try:
        info = mt5.symbol_info(symbol)
        if info is None or not getattr(info, "visible", False):
            mt5.symbol_select(symbol, True)
    except Exception:
        try:
            mt5.symbol_select(symbol, True)
        except Exception:
            pass

# ---------------------------
# entrypoint
# ---------------------------
def main():
    if not mt5.initialize():
        logger.critical("MT5 initialize failed. Ensure MT5 terminal open and logged-in.")
        return
    try:
        raw = getattr(config, "PROXY_SYMBOLS", ["VALE3","PETR4","ITUB4","BBDC4","BBAS3","WEGE3","ABEV3","MGLU3","JBSS3","RENT3"])
        symbols = select_symbols_with_sector_limits(raw)[:MAX_SYMBOLS]

        # ensure visible
        for s in symbols:
            try:
                ensure_symbol_visible(s)
            except Exception:
                pass

        # start slow loop thread
        slow_t = threading.Thread(target=slow_loop, args=(symbols,), daemon=True)
        slow_t.start()

        # run fast loop (trading enabled) in main thread
        fast_loop(symbols)

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt, exiting")
    finally:
        try:
            mt5.shutdown()
        except Exception:
            pass

if __name__ == "__main__":
    main()