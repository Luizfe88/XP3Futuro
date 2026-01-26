# bot.py — XP3 BOT B3 (CONSOLIDADO FINAL)

import time
import threading
import logging
from datetime import datetime, date
from threading import Lock
from collections import deque
import MetaTrader5 as mt5
import config
import utils

# ===== ANSI COLORS =====
C_RESET = "\033[0m"
C_GREEN = "\033[92m"
C_RED = "\033[91m"
C_YELLOW = "\033[93m"
C_CYAN = "\033[96m"
C_BOLD = "\033[1m"
C_DIM = "\033[2m"
C_WHITE = "\033[97m"


# =========================
# LOG
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("bot")

# =========================
# TIMEFRAMES
# =========================
TIMEFRAME_BASE = mt5.TIMEFRAME_M15
TIMEFRAME_FALLBACK = mt5.TIMEFRAME_M1
current_timeframe = TIMEFRAME_BASE

# =========================
# ESTADO GLOBAL
# =========================
current_top15 = []
current_indicators = {}
top15_lock = Lock()
alerts = deque(maxlen=5)   # últimos alertas
alerts_lock = Lock()
_symbol_state = {}
trading_paused = False
daily_max_equity = 0.0
last_reset_day: date | None = None
daily_slippage_atr = 0.0
last_trade_day = None

daily_equity_high = None

def push_alert(msg, level="INFO", sound=False):
    timestamp = datetime.now().strftime("%H:%M:%S")
    entry = f"[{timestamp}] {msg}"

    with alerts_lock:
        alerts.appendleft((level, entry))

    if sound:
        try:
            print("\a", end="")  # beep simples
        except Exception:
            pass

    if level == "CRITICAL":
        logger.critical(msg)
    elif level == "WARNING":
        logger.warning(msg)
    else:
        logger.info(msg)


# =========================
# BUILD TOP15 (LOOP LENTO)
# =========================
def build_portfolio_and_top15():
    scored = []
    indicators = {}

    for sym in config.SECTOR_MAP.keys():
        df = utils.safe_copy_rates(sym, TIMEFRAME_BASE, 300)
        if df is None:
            continue

        ind = utils.quick_indicators(sym, TIMEFRAME_BASE, df=df)
        if ind.get("error"):
            continue

        score = 0
        if ind["ema_fast"] > ind["ema_slow"]:
            score += 50
        if 35 <= ind["rsi"] <= 70:
            score += 30

        scored.append((score, sym))
        indicators[sym] = ind

    scored.sort(reverse=True)
    return indicators, [s for _, s in scored[:15]]

# =========================
# CIRCUIT BREAKER (DRAWDOWN)
# =========================
def check_for_circuit_breaker():
    """
    Circuit Breaker Diário baseado no maior equity do dia (intraday high watermark).
    """
    global trading_paused, daily_max_equity, last_reset_day

    acc = mt5.account_info()
    if not acc:
        return

    now = datetime.now()
    today = now.date()

    # ===========================
    # RESET DIÁRIO (APÓS HORÁRIO DEFINIDO)
    # ===========================
    reset_hour, reset_min = map(int, config.DAILY_RESET_TIME.split(":"))
    reset_time_reached = (now.hour, now.minute) >= (reset_hour, reset_min)

    if reset_time_reached and last_reset_day != today:
        daily_max_equity = acc.equity
        trading_paused = False
        last_reset_day = today

        logger.warning(
            f"🔄 RESET DIÁRIO DO CIRCUIT BREAKER | Equity base: {daily_max_equity:,.2f}"
        )
        return

    # Se ainda não inicializou (primeira execução do dia)
    if daily_max_equity == 0.0:
        daily_max_equity = acc.equity
        last_reset_day = today
        return

    # ===========================
    # ATUALIZA TOPO DE EQUITY
    # ===========================
    if acc.equity > daily_max_equity:
        daily_max_equity = acc.equity

    # ===========================
    # VERIFICA DRAWDOWN
    # ===========================
    drawdown_pct = (daily_max_equity - acc.equity) / daily_max_equity

    if drawdown_pct >= config.MAX_DAILY_DRAWDOWN_PCT:
        if not trading_paused:
            trading_paused = True
            logger.critical(
                f"🚨 CIRCUIT BREAKER ATIVADO | "
                f"DD: {drawdown_pct*100:.2f}% | "
                f"Topo: {daily_max_equity:,.2f} | "
                f"Atual: {acc.equity:,.2f}"
            )


# =========================
# SLIPPAGE KILL SWITCH
# =========================
def update_daily_slippage(symbol, expected, executed):
    global daily_slippage_atr, trading_paused, last_trade_day

    today = datetime.now().date()
    if last_trade_day != today:
        daily_slippage_atr = 0.0
        last_trade_day = today

    atr = current_indicators.get(symbol, {}).get("atr")
    if not atr or atr <= 0:
        return

    daily_slippage_atr += abs(executed - expected) / atr

    max_slip = getattr(config, "MAX_DAILY_SLIPPAGE_ATR", None)
    if max_slip and daily_slippage_atr >= max_slip:
        trading_paused = True
        logger.critical(
            f"🛑 SLIPPAGE KILL SWITCH | ATR acumulado {daily_slippage_atr:.2f}"
        )

# =========================
# TRAILING STOP
# =========================
def manage_trailing_stops():
    positions = mt5.positions_get()
    if not positions:
        return

    for p in positions:
        atr = current_indicators.get(p.symbol, {}).get("atr")
        if not atr:
            continue

        tick = mt5.symbol_info_tick(p.symbol)
        price = tick.bid if p.type == mt5.ORDER_TYPE_BUY else tick.ask
        move = abs(price - p.price_open)

        if move >= 3.0 * atr:
            new_sl = price - atr if p.type == mt5.ORDER_TYPE_BUY else price + atr
        elif move >= 1.5 * atr:
            new_sl = p.price_open
        else:
            continue

        push_alert(
            f"🔄 SL MOVIDO {p.symbol} → {new_sl:.2f}",
            level="WARNING",
            sound=False
        )

        mt5.order_send({
            "action": mt5.TRADE_ACTION_SLTP,
            "position": p.ticket,
            "sl": new_sl,
            "tp": p.tp
        })

# =========================
# FECHAMENTO SEXTA-FEIRA
# =========================
def close_positions_before_weekend():
    now = datetime.now()
    if now.weekday() != 4:
        return

    if now.strftime("%H:%M") < config.TRADING_END:
        return

    positions = mt5.positions_get()
    if not positions:
        return

    logger.warning("⚠️ Fechando posições antes do fim de semana")

    for p in positions:
        tick = mt5.symbol_info_tick(p.symbol)
        if not tick:
            continue

        order_type = (
            mt5.ORDER_TYPE_SELL if p.type == mt5.ORDER_TYPE_BUY
            else mt5.ORDER_TYPE_BUY
        )
        price = tick.bid if order_type == mt5.ORDER_TYPE_SELL else tick.ask

        mt5.order_send({
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": p.symbol,
            "volume": p.volume,
            "type": order_type,
            "price": price,
            "magic": p.magic,
            "comment": "Weekend risk close"
        })

# =========================
# FAST PROCESS
# =========================
def process_symbol_fast(symbol):
    global trading_paused

    # ===========================
    # CIRCUIT BREAKER
    # ===========================
    if trading_paused:
        return

    # ===========================
    # LIMITE DE POSIÇÕES ABERTAS
    # ===========================
    total_positions = mt5.positions_total()
    if total_positions >= config.MAX_SYMBOLS:
        logger.warning(
            f"⛔ LIMITE DE POSIÇÕES ATINGIDO "
            f"({total_positions}/{config.MAX_SYMBOLS})"
        )
        return

    ind = current_indicators.get(symbol)
    if not ind:
        return

    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        return

    price = tick.ask

    # ===========================
    # EXEMPLO DE SINAL (simplificado)
    # ===========================
    ema_ok = ind["ema_fast"] > ind["ema_slow"]
    rsi_ok = 35 <= ind["rsi"] <= 70

    df = utils.safe_copy_rates(symbol, current_timeframe, 50)
    vwap = utils.get_intraday_vwap(df)

    if not (ema_ok and rsi_ok):
        return

    if vwap and price <= vwap:
        return

    # ===========================
    # ENVIO DE ORDEM
    # ===========================
    atr = ind["atr"]
    sl = price - atr * config.SL_ATR_MULT
    tp = price + atr * config.TP_ATR_MULT

    volume = utils.calculate_position_size(symbol, sl)
    if not volume:
        return

    res = utils.send_order_with_sl_tp(symbol, "BUY", volume, sl, tp)

    if res and res.retcode == mt5.TRADE_RETCODE_DONE:
        logger.info(
            f"🟢 COMPRA EXECUTADA | {symbol} | "
            f"Vol: {volume} | Preço: {price:.2f}"
        )


# =========================
# FAST LOOP
# =========================
def fast_loop():
    logger.info("⚡ FAST LOOP INICIADO")

    while True:
        try:
            check_for_circuit_breaker()

            with top15_lock:
                symbols = list(current_top15)

            for sym in symbols:
                process_symbol_fast(sym)

            render_panel()
            time.sleep(config.FAST_LOOP_INTERVAL_SECONDS)

        except Exception:
            logger.exception("❌ Erro no FAST LOOP")
            time.sleep(5)


def render_panel():
    clear_screen()

    acc = mt5.account_info()
    if not acc:
        print("Sem dados da conta")
        return

    now = datetime.now().strftime("%H:%M:%S")
    pnl = acc.equity - acc.balance
    pnl_color = C_GREEN if pnl >= 0 else C_RED
    status = "ATIVO" if not trading_paused else "PAUSADO"
    status_color = C_GREEN if not trading_paused else C_RED

    print(f"╔{box_line()}╗")
    print(box_row(f"🚀 XP3 FAST BOT – B3    ⏱ {now}"))
    print(f"╠{box_line()}╣")

    print(box_row(
        f"Status: {status_color}{status}{C_RESET} | "
        f"Circuit: {'🟢 OK' if not trading_paused else '🔴 PAUSADO'} | "
        f"Risco: {utils.get_current_risk_pct()*100:.2f}%"
    ))

    print(box_row(
        f"Equity: R$ {money(acc.equity)} | "
        f"Balance: R$ {money(acc.balance)} | "
        f"PnL: {pnl_color}{money(pnl)}{C_RESET}"
    ))

    print(f"╠{box_line()}╣")
    print(box_row("📌 POSIÇÕES ABERTAS"))

    positions = mt5.positions_get()
    if not positions:
        print(box_row("  (nenhuma posição aberta)"))
    else:
        for p in positions:
            side = "BUY" if p.type == mt5.ORDER_TYPE_BUY else "SELL"
            color = C_GREEN if p.profit >= 0 else C_RED
            line = (
                f"{p.symbol:6} {side:4} "
                f"{p.price_open:.2f} → {p.price_current:.2f} "
                f"PnL: {color}{money(p.profit)}{C_RESET}"
            )
            print(box_row(line))

    print(f"╠{box_line()}╣")
    print(box_row("📊 TOP15 – DECISÃO OPERACIONAL"))
    print(box_row("SYM   EMA   RSI   ATR    VWAP    DECISÃO    MOTIVO"))

    with top15_lock:
        symbols = list(current_top15)

    for sym in symbols:
        ind = current_indicators.get(sym)
        if not ind:
            continue

        ema_ok = ind["ema_fast"] > ind["ema_slow"]
        rsi_ok = 35 <= ind["rsi"] <= 70

        tick = mt5.symbol_info_tick(sym)
        if not tick:
            continue

        price = (tick.bid + tick.ask) / 2
        df = utils.safe_copy_rates(sym, current_timeframe, 50)
        vwap = utils.get_intraday_vwap(df)
        vwap_txt = f"{vwap:.2f}" if vwap else "--"

        decision = ema_ok and rsi_ok and (not vwap or price > vwap)
        decision_txt = "🟢 COMPRAR" if decision else "🔴 BLOQ."
        reason = "OK"
        if not ema_ok:
            reason = "EMA"
        elif not rsi_ok:
            reason = "RSI"
        elif vwap and price <= vwap:
            reason = "VWAP"

        line = (
            f"{sym:5} "
            f"{'🟢' if ema_ok else '🔴'} "
            f"{ind['rsi']:5.1f} "
            f"{ind['atr']:6.2f} "
            f"{vwap_txt:>6} "
            f"{decision_txt:10} "
            f"{reason}"
        )

        print(box_row(line))

    print(f"╚{box_line()}╝")



def clear_screen():
    import os
    os.system("cls" if os.name == "nt" else "clear")


def line(char="═", width=78):
    return char * width

def color_bool(ok):
    return f"{C_GREEN}🟢{C_RESET}" if ok else f"{C_RED}🔴{C_RESET}"

PANEL_WIDTH = 78

def box_line(char="═"):
    return char * PANEL_WIDTH

def box_row(text=""):
    clean = text[:PANEL_WIDTH]
    return f"║ {clean.ljust(PANEL_WIDTH - 2)}║"

def money(v):
    return f"{v:,.2f}"

def panel_loop():
    while True:
        try:
            render_panel()
            time.sleep(5)
        except Exception as e:
            logger.error(f"Erro no painel: {e}")
            time.sleep(5)

# =========================
# MAIN
# =========================
def main():
    logger.info("🚀 XP3 BOT INICIANDO")
    
    if not mt5.initialize():
        logger.critical("❌ MT5 não inicializou")
        return

    ind, top = build_portfolio_and_top15()
    with top15_lock:
        global current_indicators, current_top15
        current_indicators = ind
        current_top15 = top

    threading.Thread(target=panel_loop, daemon=True).start()

    while True:
        time.sleep(3600)
        ind, top = build_portfolio_and_top15()
        with top15_lock:
            current_indicators = ind
            current_top15 = top
        logger.info(f"TOP15 atualizado → {top}")

if __name__ == "__main__":
    main()
