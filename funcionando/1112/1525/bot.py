# bot.py — versão atualizada com correções do painel, equity fallback, AUTO hybrid, e integração utils.load_optimized_summaries
import os
import time
import threading
import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
import MetaTrader5 as mt5
import pandas as pd
import numpy as np

# colors (colorama if available)
try:
    from colorama import init as colorama_init, Fore, Style
    colorama_init(autoreset=True)
    C_GREEN = Fore.GREEN; C_RED = Fore.RED; C_YELLOW = Fore.YELLOW; C_CYAN = Fore.CYAN; C_MAGENTA = Fore.MAGENTA; C_RESET = Style.RESET_ALL; C_BOLD = Style.BRIGHT
except Exception:
    C_GREEN = "\033[92m"; C_RED = "\033[91m"; C_YELLOW = "\033[93m"; C_CYAN = "\033[96m"; C_MAGENTA = "\033[95m"; C_RESET = "\033[0m"; C_BOLD = "\033[1m"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("bot")

# load config & utils
try:
    import config
except Exception:
    config = None

try:
    import utils
except Exception:
    utils = None

# timeframe mapping (strings to MT5 constants)
TF_MAP = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "H1": mt5.TIMEFRAME_H1,
    "D1": mt5.TIMEFRAME_D1
}
TF_STR = getattr(config, "TIMEFRAME_DEFAULT", "M15") if config else "M15"
TIMEFRAME_MT5 = TF_MAP.get(TF_STR, mt5.TIMEFRAME_M15)
FAST_INTERVAL_SEC = getattr(config, "FAST_LOOP_INTERVAL_SECONDS", 1.0) if config else 1.0
SLOW_INTERVAL_SEC = getattr(config, "SCAN_INTERVAL_SECONDS", 60) if config else 60

# defaults
DEFAULT_PARAMS = getattr(config, "DEFAULT_PARAMS", {
    "ema_short": 9,
    "ema_long": 21,
    "adx_threshold": 20
})

# trailing/atr
SL_ATR_MULT = getattr(config, "SL_ATR_MULT", 2.0) if config else 2.0
TP_ATR_MULT = getattr(config, "TP_ATR_MULT", 3.0) if config else 3.0
TRAILING_STEP_ATR_MULTIPLIER = getattr(config, "TRAILING_STEP_ATR_MULTIPLIER", 1.0) if config else 1.0

# circuit breaker
MAX_DAILY_LOSS_BRL = getattr(config, "MAX_DAILY_LOSS_BRL", 15000.0) if config else 15000.0
CB_CLOSE_POSITIONS = getattr(config, "CB_CLOSE_POSITIONS", True) if config else True

# state
_symbol_state: Dict[str, Dict[str, Any]] = {}
_recent_orders: Dict[str, float] = {}
optimizer_cache: Dict[str, Dict[str, Any]] = {}
circuit_breaker_tripped = False
starting_equity = None
_persisted_state: Dict[str, Any] = {}
_current_daily_realized_profit = 0.0

# helpers
def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")

def fmt(x, prec=4):
    if x is None:
        return "N/A"
    try:
        return f"{float(x):.{prec}f}"
    except Exception:
        return str(x)

def get_equity_fallback():
    try:
        info = mt5.account_info()
        if info:
            eq = float(getattr(info, "equity", 0.0) or 0.0)
            if eq and eq > 0:
                return eq
            bal = float(getattr(info, "balance", 0.0) or 0.0)
            return bal
    except Exception:
        pass
    return 0.0

# load optimizer targets per symbol (uses optimizer_cache)
def load_optimizer_targets(symbol: str) -> Optional[dict]:
    global optimizer_cache
    if optimizer_cache and symbol in optimizer_cache:
        p = optimizer_cache[symbol]
        mapped = {
            "ema_fast": int(p.get("ema_short", DEFAULT_PARAMS.get("ema_short", 9))),
            "ema_slow": int(p.get("ema_long", DEFAULT_PARAMS.get("ema_long", 21))),
            "rsi_low": float(p.get("rsi_low", 30)),
            "rsi_high": float(p.get("rsi_high", 70)),
            "mom_min": float(p.get("mom_min", 0.0)),
            "adx_period": int(p.get("adx_period", 14)),
            "adx_threshold": float(p.get("adx_threshold", DEFAULT_PARAMS.get("adx_threshold", 20)))
        }
        return mapped
    # fallback: return defaults so bot can still evaluate
    return {
        "ema_fast": DEFAULT_PARAMS.get("ema_short", 9),
        "ema_slow": DEFAULT_PARAMS.get("ema_long", 21),
        "rsi_low": 30.0,
        "rsi_high": 70.0,
        "mom_min": 0.0,
        "adx_period": 14,
        "adx_threshold": DEFAULT_PARAMS.get("adx_threshold", 20)
    }

# scoring used to build AUTO universe (hybrid)
def score_asset(ind: Dict[str, Any], targets: Optional[Dict[str, Any]] = None) -> float:
    if not ind or ind.get("error"):
        return 0.0
    score = 0.0
    adx = ind.get("adx") or 0.0
    score += min(adx, 50) * 1.2
    ema_f = ind.get("ema_fast") or 0.0
    ema_s = ind.get("ema_slow") or 0.0
    if ema_f > ema_s:
        score += 20.0
    mom = ind.get("mom") or 0.0
    score += max(min(mom * 1000, 10), -5)
    last_vol = ind.get("last_vol") or 0
    vol_mean = ind.get("vol_mean") or 1
    if vol_mean > 0 and last_vol >= vol_mean:
        score += 8.0
    else:
        ratio = (last_vol / vol_mean) if vol_mean > 0 else 0.0
        score += max(min(ratio * 4.0, 4.0), -2.0)
    rsi = ind.get("rsi")
    if rsi is not None:
        if 40 <= rsi <= 65:
            score += 6.0
        else:
            if rsi > 80 or rsi < 20:
                score -= 4.0
    if targets:
        t_fast = targets.get("ema_fast")
        t_slow = targets.get("ema_slow")
        if t_fast and t_slow and t_fast < t_slow and (ema_f > ema_s):
            score += 10.0
    score = max(min(score, 100.0), 0.0)
    return float(score)

def build_auto_universe(mode: str = "hybrid", top_n: int = 15, workers: int = 8, lookback: int = 300) -> List[str]:
    logger.info(f"{C_CYAN}Building AUTO universe (mode={mode}, top_n={top_n}){C_RESET}")
    universe = list(getattr(config, "SECTOR_MAP", {}).keys()) if config else []
    if not universe:
        universe = list(getattr(config, "PROXY_SYMBOLS", []))
    # scan in parallel
    inds = {}
    if utils and hasattr(utils, "scan_universe"):
        inds = utils.scan_universe(universe, TIMEFRAME_MT5, lookback=lookback, workers=workers)
    else:
        for s in universe:
            try:
                inds[s] = utils.quick_indicators(s, TIMEFRAME_MT5, lookback) if utils and hasattr(utils, "quick_indicators") else {"symbol": s, "error": "no_utils"}
            except Exception:
                inds[s] = {"symbol": s, "error": "exception"}
    # ensure optimizer cache loaded
    global optimizer_cache
    base_dir = getattr(config, "OPTIMIZER_OUTPUT", "optimizer_output") if config else "optimizer_output"
    if (not optimizer_cache) and utils and hasattr(utils, "load_optimized_summaries"):
        try:
            optimizer_cache = utils.load_optimized_summaries(universe, base_dir)
        except Exception:
            optimizer_cache = {}
    scored = []
    for s, ind in inds.items():
        targets = optimizer_cache.get(s) if optimizer_cache else None
        sc = score_asset(ind, targets)
        scored.append((s, sc, ind))
    scored.sort(key=lambda x: x[1], reverse=True)
    selected = []
    prefer = list(getattr(config, "PROXY_SYMBOLS", []))
    # prioritize preferred that show strong trend
    for sym in prefer:
        row = next((r for r in scored if r[0] == sym), None)
        if not row:
            continue
        ind = row[2]
        adx = ind.get("adx") or 0.0
        ema_f = ind.get("ema_fast") or 0.0
        ema_s = ind.get("ema_slow") or 0.0
        adx_th = getattr(config, "DEFAULT_PARAMS", {}).get("adx_threshold", 20)
        if adx >= adx_th and ema_f > ema_s:
            selected.append(sym)
            if len(selected) >= top_n:
                return selected[:top_n]
    # fill with top-scored
    for sym, sc, ind in scored:
        if sym in selected:
            continue
        selected.append(sym)
        if len(selected) >= top_n:
            break
    if mode.lower() == "mixed":
        for sym in prefer:
            if sym not in selected:
                selected.append(sym)
                if len(selected) >= top_n:
                    break
    return selected[:top_n]

# init symbol state (keeps EMA etc.)
def init_symbol_state(symbol: str):
    try:
        df = None
        if utils and hasattr(utils, "safe_copy_rates"):
            df = utils.safe_copy_rates(symbol, TIMEFRAME_MT5, 500)
        if df is None or df.empty:
            return
        close = df["close"].astype(float)
        ema_fast = float(close.ewm(span=DEFAULT_PARAMS.get("ema_short", 9), adjust=False).mean().iloc[-1])
        ema_slow = float(close.ewm(span=DEFAULT_PARAMS.get("ema_long", 21), adjust=False).mean().iloc[-1])
        ema_fast_prev = float(close.ewm(span=DEFAULT_PARAMS.get("ema_short", 9), adjust=False).mean().iloc[-2]) if len(close) > 1 else ema_fast
        ema_slow_prev = float(close.ewm(span=DEFAULT_PARAMS.get("ema_long", 21), adjust=False).mean().iloc[-2]) if len(close) > 1 else ema_slow
        _symbol_state[symbol] = {
            "ema_fast": ema_fast,
            "ema_slow": ema_slow,
            "ema_fast_prev": ema_fast_prev,
            "ema_slow_prev": ema_slow_prev,
            "last_price": float(close.iloc[-1]),
            "targets": load_optimizer_targets(symbol)
        }
    except Exception:
        logger.exception(f"init_symbol_state error {symbol}")

def update_ema_incremental(symbol: str, price: float):
    s = _symbol_state.get(symbol)
    if not s:
        return
    try:
        fast_n = s.get("targets", {}).get("ema_fast", DEFAULT_PARAMS.get("ema_short", 9))
        slow_n = s.get("targets", {}).get("ema_slow", DEFAULT_PARAMS.get("ema_long", 21))
        af = 2.0 / (fast_n + 1.0)
        aslow = 2.0 / (slow_n + 1.0)
        prev_fast = s.get("ema_fast", price)
        prev_slow = s.get("ema_slow", price)
        s["ema_fast_prev"] = prev_fast
        s["ema_slow_prev"] = prev_slow
        s["ema_fast"] = af * price + (1 - af) * prev_fast
        s["ema_slow"] = aslow * price + (1 - aslow) * prev_slow
        s["last_price"] = price
        _symbol_state[symbol] = s
    except Exception:
        logger.exception(f"update_ema_incremental error {symbol}")

def indicators_snapshot_with_diagnostics(symbol: str, timeframe, lookback=300):
    if utils and hasattr(utils, "quick_indicators"):
        return utils.quick_indicators(symbol, timeframe, lookback)
    return {"symbol": symbol, "error": "no_quick"}

# trailing adjust (keeps existing semantics)
def check_and_adjust_sl(position, current_price: float, atr: float, params: Dict[str, Any]):
    try:
        if position is None:
            return False
        import MetaTrader5 as mt5
        side_buy = (position.type == mt5.ORDER_TYPE_BUY)
        current_sl = float(getattr(position, "sl", 0.0) or 0.0)
        price_open = float(getattr(position, "price_open", 0.0) or 0.0)
        if atr is None or atr <= 0:
            return False
        step = TRAILING_STEP_ATR_MULTIPLIER * atr
        if side_buy:
            new_sl = float(max(0.0001, current_price - SL_ATR_MULT * atr))
            if new_sl <= current_sl:
                return False
            if new_sl <= price_open:
                new_sl = max(new_sl, price_open)
            if current_sl != 0.0 and (new_sl - current_sl) < step:
                return False
        else:
            new_sl = float(current_price + SL_ATR_MULT * atr)
            if new_sl >= current_sl and current_sl != 0.0:
                return False
            if current_sl != 0.0 and (current_sl - new_sl) < step:
                return False
        ticket = getattr(position, "ticket", None)
        if ticket is None:
            return False
        try:
            price_field = float(getattr(position, "price_open", getattr(position, "price", 0.0)) or 0.0)
            tp_field = float(getattr(position, "tp", 0.0) or 0.0)
            res = mt5.order_modify(int(ticket), price_field, float(new_sl), float(tp_field))
            retcode = getattr(res, "retcode", None)
            if retcode in (mt5.TRADE_RETCODE_DONE, 10009):
                logger.info(f"Modified SL for {position.symbol}: {current_sl:.6f} -> {new_sl:.6f}")
                return True
            else:
                logger.warning(f"order_modify failed for {position.symbol}: ret={retcode} res={res}")
                return False
        except Exception as e:
            logger.exception(f"check_and_adjust_sl modify error for {position.symbol}: {e}")
            return False
    except Exception as e:
        logger.exception(f"check_and_adjust_sl generic error: {e}")
        return False

# render panel (formatted)
def render_panel(equity: float, portfolio_indicators: Dict[str, Dict[str,Any]], positions: List[Dict[str,Any]]):
    clear_screen()
    now = datetime.now()
    now_str = now.strftime("%Y-%m-%d %H:%M:%S")
    print(C_BOLD + C_MAGENTA + "===== FAST BOT PANEL =====" + C_RESET)
    print(f"{C_YELLOW}Relógio: {now_str}{C_RESET}    Fast loop: {FAST_INTERVAL_SEC:.1f}s    Slow loop: {SLOW_INTERVAL_SEC}s")
    if equity is None or equity <= 0:
        print(C_RED + "AVISO: Equity retornou 0.00 — verifique MT5." + C_RESET)
    print(f"Equity: {C_BOLD}{C_YELLOW}{equity:,.2f}{C_RESET}\n")
    print(C_BOLD + C_CYAN + "=== POSIÇÕES EM CARTEIRA ===" + C_RESET)
    if not positions:
        print("  (nenhuma posição aberta)")
    else:
        for p in positions:
            prof = p.get("profit", 0.0)
            prof_col = C_GREEN if prof >= 0 else C_RED
            vol = p.get("volume", 0.0)
            price = p.get("price", 0.0)
            sl = p.get("sl", 0.0)
            print(f"  {p['symbol']:6s} Qty:{vol:8.2f} Price:{price:12.2f} SL:{sl:.4f} Profit:{prof_col}{prof:10.2f}{C_RESET}")
    print("")
    hdr = f"{'SYM':6s} {'EMA (f>s)':18s} {'RSI':6s} {'MOM':9s} {'ATR':8s} {'ADX':6s} {'TH':4s} {'VOL(L/M)':12s} {'MYQTY':6s} {'STATUS':6s} {'OPT_TARGETS'}"
    print(C_BOLD + C_MAGENTA + hdr + C_RESET)
    for sym, d in portfolio_indicators.items():
        if not d or d.get("error"):
            err = d.get("error") if isinstance(d, dict) else "no_data"
            print(f"{sym:6s} {C_RED}ERROR:{err}{C_RESET}")
            continue
        targets = d.get("targets") or load_optimizer_targets(sym)
        ema_f = d.get("ema_fast"); ema_s = d.get("ema_slow")
        rsi = d.get("rsi"); mom = d.get("mom"); atr = d.get("atr"); adx = d.get("adx")
        last_vol = d.get("last_vol", 0); vol_mean = d.get("vol_mean", 0)
        adx_th = targets.get("adx_threshold", DEFAULT_PARAMS.get("adx_threshold", 20))
        ema_ok = (ema_f is not None and ema_s is not None and ema_f > ema_s)
        adx_ok = (adx is not None and adx >= adx_th)
        status_ok = ema_ok and adx_ok
        # my qty
        my_qty = 0.0
        try:
            pos = mt5.positions_get(symbol=sym)
            if pos and len(pos) > 0:
                my_qty = float(pos[0].volume)
        except Exception:
            my_qty = 0.0
        # format values
        ema_str = f"{ema_f:.2f}>{ema_s:.2f}" if ema_f is not None and ema_s is not None else "N/A"
        rsi_str = f"{rsi:.1f}" if rsi is not None else "N/A"
        mom_str = f"{mom:.4f}" if mom is not None else "N/A"
        atr_str = f"{atr:.4f}" if atr is not None else "N/A"
        adx_str = f"{adx:.2f}" if adx is not None else "N/A"
        vol_str = f"{int(last_vol):d}/{int(vol_mean):d}" if vol_mean>0 else f"{int(last_vol):d}/0"
        tgt_str = f"EMA {int(targets.get('ema_fast')) if targets else '?'} / {int(targets.get('ema_slow')) if targets else '?'}"
        status_col = C_GREEN if status_ok else C_RED
        print(f"{sym:6s} {ema_str:18s} {rsi_str:6s} {mom_str:9s} {atr_str:8s} {adx_str:6s} {int(adx_th):4d} {vol_str:12s} {my_qty:6.2f} {status_col}{('OK' if status_ok else 'NO'):6s}{C_RESET} {C_CYAN}{tgt_str}{C_RESET}")
    print("\n" + C_BOLD + C_MAGENTA + "==========================" + C_RESET)

# close positions (CB)
def close_all_positions():
    try:
        pos = mt5.positions_get()
        if not pos:
            logger.info("close_all_positions: none")
            return
        for p in pos:
            try:
                sym = p.symbol
                vol = float(p.volume)
                side = "SELL" if p.type == mt5.ORDER_TYPE_BUY else "BUY"
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": sym,
                    "volume": vol,
                    "type": mt5.ORDER_TYPE_SELL if side=="SELL" else mt5.ORDER_TYPE_BUY,
                    "price": float(mt5.symbol_info_tick(sym).bid if side=="SELL" else mt5.symbol_info_tick(sym).ask),
                    "magic": 99,
                    "comment": "cb_close",
                    "type_filling": mt5.ORDER_FILLING_IOC
                }
                res = mt5.order_send(request)
                logger.info(f"close_all_positions: closed {sym} vol={vol} -> {getattr(res,'retcode',str(res))}")
            except Exception:
                logger.exception("Error closing position")
    except Exception:
        logger.exception("Error in close_all_positions")

def in_market_hours():
    """
    Retorna True se o horário atual estiver dentro do horário de pregão definido no config.py.
    """
    try:
        import config
        now = datetime.now()

        open_h = getattr(config, "MARKET_OPEN_HOUR", 10)
        open_m = getattr(config, "MARKET_OPEN_MINUTE", 0)
        close_h = getattr(config, "MARKET_CLOSE_HOUR", 17)
        close_m = getattr(config, "MARKET_CLOSE_MINUTE", 0)

        open_t = now.replace(hour=open_h, minute=open_m, second=0, microsecond=0)
        close_t = now.replace(hour=close_h, minute=close_m, second=0, microsecond=0)

        return open_t <= now <= close_t

    except Exception:
        # fallback: sempre True se der problema
        return True


# fast loop (keeps trading)
def fast_loop(symbols: List[str]):
    logger.info("Fast loop started")
    for s in symbols:
        init_symbol_state(s)
    while True:
        try:
            for sym in symbols:
                if circuit_breaker_tripped:
                    continue
                if not in_market_hours():
                    continue
                tick = mt5.symbol_info_tick(sym)
                if tick is None:
                    continue
                price = float(tick.ask) if getattr(tick, "ask", None) and tick.ask>0 else float(tick.bid)
                if sym not in _symbol_state:
                    init_symbol_state(sym)
                    continue
                update_ema_incremental(sym, price)
                state = _symbol_state[sym]
                targets = state.get("targets") or load_optimizer_targets(sym)
                # crossover detection (simple)
                prev_fast = state.get("ema_fast_prev"); prev_slow = state.get("ema_slow_prev")
                cur_fast = state.get("ema_fast"); cur_slow = state.get("ema_slow")
                if None in (prev_fast, prev_slow, cur_fast, cur_slow):
                    continue
                is_cross = (cur_fast > cur_slow and prev_fast <= prev_slow)
                # check other filters (rsi, mom, adx)
                df = utils.safe_copy_rates(sym, TIMEFRAME_MT5, 300) if utils and hasattr(utils, "safe_copy_rates") else None
                if df is None:
                    continue
                # compute quick metrics
                rsi = None; mom = None; atr = None; adx_val = None
                try:
                    delta = df['close'].astype(float).diff()
                    up = delta.clip(lower=0).rolling(14).mean()
                    down = -delta.clip(upper=0).rolling(14).mean()
                    rs = up / down
                    rsi = float((100 - (100 / (1 + rs))).iloc[-1])
                except Exception:
                    rsi = None
                try:
                    mom = float(df['close'].astype(float).iloc[-1] / df['close'].astype(float).shift(10).iloc[-1] - 1) if len(df)>10 else 0.0
                except Exception:
                    mom = None
                try:
                    atr = utils.get_atr(df, 14) if utils and hasattr(utils, "get_atr") else None
                except Exception:
                    atr = None
                # adx via pandas_ta fallback
                try:
                    import pandas_ta as ta
                    adx_p = int(targets.get("adx_period", 14))
                    adx_df = ta.adx(df["high"], df["low"], df["close"], length=adx_p)
                    for c in adx_df.columns:
                        if "ADX" in c.upper():
                            adx_val = float(adx_df[c].iloc[-1]); break
                except Exception:
                    if "ADX_14" in df.columns:
                        adx_val = float(df["ADX_14"].iloc[-1])
                    elif "ADX" in df.columns:
                        adx_val = float(df["ADX"].iloc[-1])
                    else:
                        adx_val = None
                # simple trade decision (only example, keep existing order logic!)
                adx_th = float(targets.get("adx_threshold", DEFAULT_PARAMS.get("adx_threshold", 20)))
                rsi_ok = (rsi is not None and targets.get("rsi_low",30) <= rsi <= targets.get("rsi_high",70))
                mom_ok = (mom is not None and mom >= targets.get("mom_min", 0.0))
                adx_ok = (adx_val is not None and adx_val >= adx_th)
                trade_ok = is_cross and rsi_ok and mom_ok and adx_ok
                positions = mt5.positions_get(symbol=sym)
                has_pos = bool(positions and len(positions)>0)
                if trade_ok and not has_pos:
                    if atr is None or atr <= 0:
                        continue
                    last_price = price
                    sl_price = last_price - SL_ATR_MULT * atr
                    tp_price = last_price + TP_ATR_MULT * atr
                    lots = utils.calculate_position_size(sym, sl_price, risk_pct=0.01) if utils and hasattr(utils, "calculate_position_size") else None
                    if not lots or lots <= 0:
                        continue
                    sig = f"{sym}|BUY|{lots:.4f}|{sl_price:.4f}|{tp_price:.4f}"
                    if sig in _recent_orders and (time.time() - _recent_orders[sig] < getattr(config, "DUP_ORDER_COOLDOWN_SECONDS", 60)):
                        continue
                    res = utils.send_order_with_sl_tp(sym, "BUY", lots, sl_price, tp_price) if utils and hasattr(utils, "send_order_with_sl_tp") else {"success": False, "reason": "no_send"}
                    if res.get("success"):
                        _recent_orders[sig] = time.time()
                else:
                    # trailing adjust existing pos
                    if positions:
                        pos = positions[0]
                        check_and_adjust_sl(pos, price, atr, targets)
            time.sleep(0.01)
        except Exception:
            logger.exception("fast_loop error")
        time.sleep(FAST_INTERVAL_SEC)

# slow loop: build universe, load optimizer cache, render panel, persist
def slow_loop():
    global optimizer_cache, circuit_breaker_tripped, starting_equity, _current_daily_realized_profit
    logger.info("Slow loop started")
    PROXY_MODE = getattr(config, "PROXY_MODE", "MANUAL") if config else "MANUAL"
    PROXY_AUTO_N = getattr(config, "PROXY_AUTO_N", 15) if config else 15
    PROXY_AUTO_WORKERS = getattr(config, "PROXY_AUTO_WORKERS", min(8, (os.cpu_count() or 4))) if config else min(8, (os.cpu_count() or 4))
    if starting_equity is None:
        try:
            starting_equity = get_equity_fallback()
        except Exception:
            starting_equity = 0.0
    while True:
        try:
            # reload optimizer cache (use all universe)
            try:
                base_dir = getattr(config, "OPTIMIZER_OUTPUT", "optimizer_output")
                if not os.path.isabs(base_dir):
                    base_dir = os.path.join(os.getcwd(), base_dir)
                if utils and hasattr(utils, "load_optimized_summaries"):
                    universe = list(getattr(config, "SECTOR_MAP", {}).keys()) if config else []
                    optimizer_cache = utils.load_optimized_summaries(universe, base_dir)
                    logger.info(f"optimizer cache loaded: {len(optimizer_cache)} symbols")
            except Exception:
                logger.exception("Failed loading optimizer cache")
            # build universe
            if getattr(config, "PROXY_MODE", "MANUAL").upper() == "AUTO":
                symbols = build_auto_universe(mode="hybrid", top_n=PROXY_AUTO_N, workers=PROXY_AUTO_WORKERS, lookback=300)
            else:
                symbols = getattr(config, "PROXY_SYMBOLS", [])
            # ensure visible and init states
            selected = []
            for s in symbols:
                try:
                    info = mt5.symbol_info(s)
                    if info is None or not getattr(info, "visible", True):
                        try:
                            mt5.symbol_select(s, True)
                        except Exception:
                            pass
                    init_symbol_state(s)
                    selected.append(s)
                except Exception:
                    logger.exception(f"init failed for {s}")
            # build indicators snapshot for panel
            portfolio_indicators = {}
            for sym in selected:
                st = _symbol_state.get(sym, {})
                quick = utils.quick_indicators(sym, TIMEFRAME_MT5, lookback=300) if utils and hasattr(utils, "quick_indicators") else {}
                merged = {
                    "symbol": sym,
                    "ema_fast": quick.get("ema_fast", st.get("ema_fast")),
                    "ema_slow": quick.get("ema_slow", st.get("ema_slow")),
                    "rsi": quick.get("rsi"),
                    "mom": quick.get("mom"),
                    "atr": quick.get("atr"),
                    "adx": quick.get("adx"),
                    "last_vol": quick.get("last_vol"),
                    "vol_mean": quick.get("vol_mean"),
                    "targets": load_optimizer_targets(sym)
                }
                portfolio_indicators[sym] = merged
            # account & positions
            equity = get_equity_fallback()
            positions = []
            try:
                pos = mt5.positions_get()
                if pos:
                    for p in pos:
                        positions.append({
                            "symbol": p.symbol,
                            "volume": float(p.volume),
                            "price": float(getattr(p, "price_open", getattr(p, "price", 0.0))),
                            "sl": float(getattr(p, "sl", 0.0) or 0.0),
                            "profit": float(p.profit)
                        })
            except Exception:
                positions = []
            # circuit breaker
            if starting_equity is None:
                starting_equity = equity
            current_profit_daily = equity - starting_equity
            _current_daily_realized_profit = current_profit_daily
            if current_profit_daily < -MAX_DAILY_LOSS_BRL:
                if not circuit_breaker_tripped:
                    logger.critical(f"CIRCUIT BREAKER TRIPPED! loss {current_profit_daily:.2f}")
                    circuit_breaker_tripped = True
                    if CB_CLOSE_POSITIONS:
                        close_all_positions()
            else:
                if circuit_breaker_tripped and current_profit_daily >= -MAX_DAILY_LOSS_BRL:
                    circuit_breaker_tripped = False
            # render panel
            render_panel(equity, portfolio_indicators, positions)
            # persist state
            try:
                state_to_save = {
                    "daily_realized_profit": float(_current_daily_realized_profit),
                    "circuit_breaker_active": bool(circuit_breaker_tripped),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                if utils and hasattr(utils, "save_bot_state"):
                    utils.save_bot_state(state_to_save, getattr(config, "BOT_STATE_FILE", "bot_state.json"))
                else:
                    fp = getattr(config, "BOT_STATE_FILE", "bot_state.json")
                    with open(fp, "w", encoding="utf-8") as f:
                        json.dump(state_to_save, f, indent=2, ensure_ascii=False, default=str)
            except Exception:
                logger.exception("Error saving bot state")
        except Exception:
            logger.exception("slow_loop error")
        time.sleep(SLOW_INTERVAL_SEC)

# main
def main():
    global _persisted_state, _current_daily_realized_profit, starting_equity, optimizer_cache
    if not mt5.initialize():
        logger.critical("MT5 initialize failed. Ensure MT5 terminal open.")
        return
    # load state
    try:
        if utils and hasattr(utils, "load_bot_state"):
            _persisted_state = utils.load_bot_state(getattr(config, "BOT_STATE_FILE", "bot_state.json"))
        else:
            fp = getattr(config, "BOT_STATE_FILE", "bot_state.json")
            _persisted_state = {}
            if os.path.exists(fp):
                try:
                    _persisted_state = json.load(open(fp, "r", encoding="utf-8")) or {}
                except Exception:
                    _persisted_state = {}
        _current_daily_realized_profit = float(_persisted_state.get("daily_realized_profit", 0.0))
        starting_equity = get_equity_fallback()
    except Exception:
        _persisted_state = {}
        _current_daily_realized_profit = 0.0
    # initial universe
    PROXY_MODE = getattr(config, "PROXY_MODE", "MANUAL") if config else "MANUAL"
    if PROXY_MODE.upper() == "AUTO":
        symbols = build_auto_universe(mode="hybrid", top_n=getattr(config, "PROXY_AUTO_N", 15), workers=getattr(config, "PROXY_AUTO_WORKERS", min(8, (os.cpu_count() or 4))), lookback=300)
    else:
        symbols = getattr(config, "PROXY_SYMBOLS", [])
    # ensure visible
    for s in symbols:
        try:
            info = mt5.symbol_info(s)
            if info is None or not getattr(info, "visible", False):
                mt5.symbol_select(s, True)
        except Exception:
            pass
    # start slow loop thread
    slow_t = threading.Thread(target=slow_loop, daemon=True)
    slow_t.start()
    # run fast loop
    fast_loop(symbols)

if __name__ == "__main__":
    main()
