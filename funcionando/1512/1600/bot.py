# bot.py — XP3 BOT B3 (VERSÃO PROFISSIONAL ATUALIZADA - 15/12/2025)

import time
import threading
import logging
from datetime import datetime, date, timedelta
from threading import Lock
from collections import deque, defaultdict
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
C_BG_GREEN = "\033[102m"
C_BG_RED = "\033[101m"

def clear_screen():
    import os
    os.system("cls" if os.name == "nt" else "clear")
# =========================
# LOG
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("bot")
# Salva log em arquivo (além do console)
handler = logging.FileHandler("xp3_bot.log")
handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
logging.getLogger().addHandler(handler)

# =========================
# TIMEFRAMES
# =========================
TIMEFRAME_BASE = mt5.TIMEFRAME_M15
TIMEFRAME_MACRO = getattr(mt5, f"TIMEFRAME_{config.MACRO_TIMEFRAME}", mt5.TIMEFRAME_H1)
current_timeframe = TIMEFRAME_BASE

# =========================
# ESTADO GLOBAL
# =========================
current_top15 = []
current_indicators = {}
optimized_params = {}  # params por símbolo carregados do config ou otimizador
top15_lock = Lock()
alerts = deque(maxlen=10)
alerts_lock = Lock()
_symbol_state = {}
trading_paused = False
daily_max_equity = 0.0
last_reset_day: date | None = None
daily_slippage_atr = 0.0
last_trade_day = None
last_failure_reason = {}  # {symbol: "motivo da última falha"}
failure_lock = Lock()

def push_alert(msg, level="INFO", sound=True):
    timestamp = datetime.now().strftime("%H:%M:%S")
    entry = f"[{timestamp}] {msg}"

    with alerts_lock:
        alerts.appendleft((level, entry))

    if sound:
        try:
            print("\a", end="")
        except Exception:
            pass

    if level == "CRITICAL":
        logger.critical(msg)
    elif level == "WARNING":
        logger.warning(msg)
    else:
        logger.info(msg)

# =========================
# CARREGAR PARÂMETROS OTIMIZADOS
# =========================
def load_optimized_params():
    global optimized_params
    optimized_params = getattr(config, "OPTIMIZED_PARAMS", {}).copy()
    # Default fallback
    for sym in config.SECTOR_MAP.keys():
        if sym not in optimized_params:
            optimized_params[sym] = {
                "ema_short": 9,
                "ema_long": 21,
                "rsi_low": 35,
                "rsi_high": 70,
            }

# =========================
# BUILD TOP15 COM PARÂMETROS OTIMIZADOS
# =========================
def build_portfolio_and_top15():
    scored = []
    indicators = {}
    load_optimized_params()

    for sym in config.SECTOR_MAP.keys():
        df = utils.safe_copy_rates(sym, TIMEFRAME_BASE, 300)
        if df is None or len(df) < 50:
            continue

        params = optimized_params.get(sym, {})
        ind = utils.quick_indicators_custom(sym, TIMEFRAME_BASE, df=df, params=params)
        if ind.get("error"):
            continue

        score = 0
        if ind["ema_fast"] > ind["ema_slow"]:
            score += 50
        if params.get("rsi_low", 35) <= ind["rsi"] <= params.get("rsi_high", 70):
            score += 30
        if ind["atr"] > 0.3:  # filtro mínimo de volatilidade
            score += 10

        scored.append((score, sym))
        indicators[sym] = ind

    scored.sort(reverse=True)
    return indicators, [s for _, s in scored[:15]]

# =========================
# FILTROS COMUNS
# =========================
def is_trading_time():
    now = datetime.now()
    start = datetime.strptime(config.TRADING_START, "%H:%M").time()
    end = datetime.strptime(config.TRADING_END, "%H:%M").time()
    current = now.time()
    return start <= current <= end

def get_sector_counts():
    positions = mt5.positions_get()
    if not positions:
        return defaultdict(int)
    counts = defaultdict(int)
    for p in positions:
        sector = config.SECTOR_MAP.get(p.symbol, "UNKNOWN")
        counts[sector] += 1
    return counts

# =========================
# FILTRO MACRO
# =========================
def macro_trend_ok(symbol: str, side: str) -> bool:
    df_macro = utils.safe_copy_rates(symbol, TIMEFRAME_MACRO, 300)
    if df_macro is None or len(df_macro) < config.MACRO_EMA_LONG:
        return False
    close = df_macro["close"]
    ema200 = close.ewm(span=config.MACRO_EMA_LONG, adjust=False).mean().iloc[-1]
    current_price = mt5.symbol_info_tick(symbol).last or close.iloc[-1]

    if side == "BUY":
        return current_price > ema200
    elif side == "SELL":
        return current_price < ema200
    return False

# =========================
# GESTÃO DE POSIÇÕES AVANÇADA
# =========================
def manage_positions_advanced():
    positions = mt5.positions_get()
    if not positions:
        return

    for pos in positions:
        sym = pos.symbol
        ind = current_indicators.get(sym)
        if not ind or not ind.get("atr"):
            continue

        atr = ind["atr"]
        tick = mt5.symbol_info_tick(sym)
        if not tick:
            continue

        current_price = tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask
        profit_in_atr = abs(current_price - pos.price_open) / atr

        # Breakeven após 1.5 ATR
        if profit_in_atr >= 1.5:
            new_sl = pos.price_open + (0.1 if pos.type == mt5.ORDER_TYPE_BUY else -0.1)  # pequeno buffer
            if (pos.type == mt5.ORDER_TYPE_BUY and pos.sl < new_sl) or \
               (pos.type == mt5.ORDER_TYPE_SELL and (pos.sl > new_sl or pos.sl == 0)):
                mt5.order_send({
                    "action": mt5.TRADE_ACTION_SLTP,
                    "position": pos.ticket,
                    "sl": new_sl,
                    "tp": pos.tp
                })

        # Partial close 50% ao atingir 2x risco (2 ATR de lucro)
        if profit_in_atr >= 2.0 and pos.volume > 0.01:  # evitar micro lotes
            close_volume = round(pos.volume * 0.5, 2)
            if close_volume >= 0.01:
                order_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
                price = tick.bid if order_type == mt5.ORDER_TYPE_SELL else tick.ask
                res = mt5.order_send({
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": sym,
                    "volume": close_volume,
                    "type": order_type,
                    "position": pos.ticket,
                    "price": price,
                    "magic": 2026,
                    "comment": "Partial TP 2xR"
                })
                if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                    push_alert(f"🎯 PARTIAL CLOSE 50% {sym} @ 2xR", sound=True)

        # Trailing existente (mantido)
        move = abs(current_price - pos.price_open)
        if move >= 3.0 * atr:
            new_sl = current_price - atr if pos.type == mt5.ORDER_TYPE_BUY else current_price + atr
        elif move >= 1.5 * atr:
            new_sl = pos.price_open
        else:
            continue

        mt5.order_send({
            "action": mt5.TRADE_ACTION_SLTP,
            "position": pos.ticket,
            "sl": new_sl,
            "tp": pos.tp
        })

# =========================
# PYRAMIDING
# =========================
def try_pyramid(symbol: str, side: str, atr: float):
    positions = [p for p in mt5.positions_get(symbol=symbol) or [] if
                 (p.type == mt5.ORDER_TYPE_BUY and side == "BUY") or
                 (p.type == mt5.ORDER_TYPE_SELL and side == "SELL")]
    if len(positions) >= 2:
        return False  # máximo 2 pernas

    if not positions:
        return True  # primeira entrada

    pos = positions[0]
    tick = mt5.symbol_info_tick(symbol)
    price = tick.ask if side == "BUY" else tick.bid
    distance = abs(price - pos.price_open)

    if distance >= config.PYRAMID_ATR_DISTANCE * atr:
        # mover SL da primeira para breakeven
        breakeven = pos.price_open
        mt5.order_send({
            "action": mt5.TRADE_ACTION_SLTP,
            "position": pos.ticket,
            "sl": breakeven,
            "tp": pos.tp
        })
        return True
    return False

# =========================
# PROCESS SYMBOL FAST (BIDIRECIONAL + TODAS MELHORIAS)
# =========================
def process_symbol_fast(symbol):
    global trading_paused

    if trading_paused or not is_trading_time():
        return

    total_positions = mt5.positions_total()
    if total_positions >= config.MAX_SYMBOLS:
        return

    sector_counts = get_sector_counts()
    sector = config.SECTOR_MAP.get(symbol, "UNKNOWN")
    if sector_counts[sector] >= config.MAX_PER_SECTOR:
        return

    ind = current_indicators.get(symbol)
    if not ind or ind.get("atr", 0) < 0.3:  # filtro mínimo volatilidade
        return

    params = optimized_params.get(symbol, {})
    rsi_low = params.get("rsi_low", 35)
    rsi_high = params.get("rsi_high", 70)

    tick = mt5.symbol_info_tick(symbol)
    if not tick or tick.ask <= 0 or tick.bid <= 0:
        return

    price = (tick.ask + tick.bid) / 2
    df = utils.safe_copy_rates(symbol, current_timeframe, 100)
    vwap = utils.get_intraday_vwap(df)

    # Determinar direção
    long_condition = (
        ind["ema_fast"] > ind["ema_slow"] and
        rsi_low <= ind["rsi"] <= rsi_high and
        (not vwap or price > vwap) and
        config.TRADE_BOTH_DIRECTIONS
    )

    short_condition = (
        ind["ema_fast"] < ind["ema_slow"] and
        rsi_low <= ind["rsi"] <= rsi_high and
        (not vwap or price < vwap) and
        config.TRADE_BOTH_DIRECTIONS
    )

    side = None
    reason = []
    if long_condition and macro_trend_ok(symbol, "BUY"):
        side = "BUY"
        reason = ["EMA up", f"RSI {ind['rsi']:.1f}", "acima VWAP", "macro OK"]
        if not try_pyramid(symbol, "BUY", ind["atr"]):
            return
    elif short_condition and macro_trend_ok(symbol, "SELL"):
        side = "SELL"
        reason = ["EMA down", f"RSI {ind['rsi']:.1f}", "abaixo VWAP", "macro OK"]
        if not try_pyramid(symbol, "SELL", ind["atr"]):
            return
    else:
        return

    atr = ind["atr"]
    sl_distance = atr * config.SL_ATR_MULT
    tp_distance = atr * config.TP_ATR_MULT

    price_entry = tick.ask if side == "BUY" else tick.bid
    sl = price_entry - sl_distance if side == "BUY" else price_entry + sl_distance
    tp = price_entry + tp_distance if side == "BUY" else price_entry - tp_distance

    risk_pct = config.PYRAMID_RISK_SPLIT[1] if mt5.positions_get(symbol=symbol) else config.PYRAMID_RISK_SPLIT[0]
    volume = utils.calculate_position_size_custom(symbol, sl, risk_pct * config.RISK_PER_TRADE_PCT)

    if not volume or volume < 0.01:
        return

    res = utils.send_order_with_sl_tp(symbol, side, volume, sl, tp)

    if res and res.retcode == mt5.TRADE_RETCODE_DONE:
        motivo = f"{side} {symbol} - {' + '.join(reason)}"
        logger.info(f"{'🟢' if side=='BUY' else '🔴'} EXECUTADO: {motivo} | Vol: {volume:.2f}")
        push_alert(f"🚨 {side} {symbol} EXECUTADO!\n{motivo}", level="INFO", sound=True)
        # Limpa falha anterior ao executar com sucesso
        with failure_lock:
            if symbol in last_failure_reason:
                del last_failure_reason[symbol]

    else:
        # CAPTURA O MOTIVO DA FALHA
        if res:
            retcode_desc = mt5.retcode_description(res.retcode) if hasattr(mt5, 'retcode_description') else str(res.retcode)
            comment = res.comment if res.comment else "Sem comentário"
            reason = f"{retcode_desc} - {comment}"
        else:
            reason = "Sem resposta do MT5 (conexão?)"

        full_reason = f"Falha ordem {side}: {reason}"
        logger.warning(full_reason)
        push_alert(f"⚠️ Falha {side} {symbol}: {reason}", level="WARNING", sound=True)

        # Salva o motivo para mostrar no dashboard
        with failure_lock:
            last_failure_reason[symbol] = reason

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
        push_alert("🔄 Reset diário do Circuit Breaker realizado", level="WARNING")
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
    drawdown_pct = (daily_max_equity - acc.equity) / daily_max_equity if daily_max_equity > 0 else 0

    if drawdown_pct >= config.MAX_DAILY_DRAWDOWN_PCT:
        if not trading_paused:
            trading_paused = True
            logger.critical(
                f"🚨 CIRCUIT BREAKER ATIVADO | "
                f"DD: {drawdown_pct*100:.2f}% | "
                f"Topo: {daily_max_equity:,.2f} | "
                f"Atual: {acc.equity:,.2f}"
            )
            push_alert("🚨 CIRCUIT BREAKER ATIVADO - Trading pausado!", level="CRITICAL", sound=True)


# =========================
# FECHAMENTO SEXTA-FEIRA
# =========================
def close_positions_before_weekend():
    now = datetime.now()
    if now.weekday() != 4:  # Não é sexta-feira
        return

    end_time = datetime.strptime(config.TRADING_END, "%H:%M").time()
    if now.time() < end_time:
        return

    positions = mt5.positions_get()
    if not positions:
        return

    logger.warning("⚠️ Fechando todas as posições antes do fim de semana")
    push_alert("⚠️ Fechando posições - Fim de semana", level="WARNING", sound=True)

    for p in positions:
        tick = mt5.symbol_info_tick(p.symbol)
        if not tick:
            continue

        order_type = (
            mt5.ORDER_TYPE_SELL if p.type == mt5.ORDER_TYPE_BUY
            else mt5.ORDER_TYPE_BUY
        )
        price = tick.bid if order_type == mt5.ORDER_TYPE_SELL else tick.ask

        res = mt5.order_send({
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": p.symbol,
            "volume": p.volume,
            "type": order_type,
            "position": p.ticket,
            "price": price,
            "magic": 2026,
            "comment": "Weekend risk close"
        })

        if res and res.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"Fechada posição {p.symbol} antes do weekend")

# =========================
# FAST LOOP
# =========================
def fast_loop():
    logger.info("⚡ FAST LOOP INICIADO (BIDIRECIONAL + PYRAMID + MACRO)")

    while True:
        try:
            check_for_circuit_breaker()
            manage_positions_advanced()
            close_positions_before_weekend()

            with top15_lock:
                symbols = list(current_top15)

            for sym in symbols:
                process_symbol_fast(sym)

            render_panel_enhanced()
            time.sleep(config.FAST_LOOP_INTERVAL_SECONDS)

        except Exception as e:
            logger.exception(f"Erro no FAST LOOP: {e}")
            time.sleep(5)

# =========================
# DASHBOARD MELHORADO (TOP15 + MOTIVOS CLAROS)
# =========================
def render_panel_enhanced():
    clear_screen()

    acc = mt5.account_info()
    if not acc:
        print("Sem conexão com MT5")
        return

    now = datetime.now().strftime("%d/%m %H:%M:%S")
    pnl = acc.equity - acc.balance
    pnl_color = C_GREEN if pnl >= 0 else C_RED
    status_color = C_GREEN if not trading_paused else C_RED
    status = "ATIVO" if not trading_paused else "PAUSADO (CB)"

    print(f"{C_BOLD}╔{'═' * 90}╗{C_RESET}")
    print(f"║ {C_CYAN}🚀 XP3 PRO BOT - B3 BRASIL{C_RESET}     📅 {now}     {status_color}{status}{C_RESET} {' ' * 20}║")
    print(f"╠{'═' * 90}╣")
    print(f"║ Equity: R$ {acc.equity:,.2f}   |   Balance: R$ {acc.balance:,.2f}   |   PnL: {pnl_color}{pnl:+,.2f}{C_RESET}          ║")
    print(f"║ Posições: {mt5.positions_total()}/{config.MAX_SYMBOLS}   |   Risco/trade: {utils.get_current_risk_pct()*100:.1f}%{' '*20}║")

    # ========================
    # ÚLTIMOS ALERTAS (COMPRAS E VENDAS)
    # ========================
    print(f"╠{'═' * 90}╣")
    print(f"║ {C_BOLD}{C_YELLOW}🚨 ÚLTIMOS ALERTAS DE OPERAÇÕES{C_RESET}{' '*50}║")
    with alerts_lock:
        recent_alerts = list(alerts)[:5]  # pega os 5 mais recentes

    if not recent_alerts:
        print(f"║   {'(nenhum alerta recente)':^86} ║")
    else:
        for level, msg in recent_alerts:
            if "COMPRA EXECUTADO" in msg or "BUY" in msg:
                color = C_GREEN
                icon = "🟢 COMPRA"
            elif "VENDA EXECUTADO" in msg or "SELL" in msg:
                color = C_RED
                icon = "🔴 VENDA"
            elif "PARTIAL" in msg:
                color = C_YELLOW
                icon = "🎯 PARTIAL"
            else:
                color = C_WHITE
                icon = "ℹ️"

            clean_msg = msg.split("] ", 1)[1] if "] " in msg else msg  # remove timestamp
            line = f"{icon} {clean_msg}"
            print(f"║ {color}{line.ljust(86)}{C_RESET} ║")

    print(f"╠{'═' * 90}╣")

    # ========================
    # CARTEIRA ATUAL
    # ========================
    print(f"║ {C_BOLD}{C_GREEN}💼 CARTEIRA ATUAL (POSIÇÕES ABERTAS){C_RESET}{' '*45}║")
    positions = mt5.positions_get()
    if not positions:
        print(f"║   {'(nenhuma posição aberta)':^86} ║")
    else:
        print(f"║ {'SYM':<6} {'DIR':<4} {'VOL':<6} {'ENTRY':<9} {'ATUAL':<9} {'PnL R$':<11} {'%':<7} {'STATUS':<18} ║")
        print(f"║ {'─'*88} ║")
        for p in positions:
            sym_info = mt5.symbol_info(p.symbol)
            if not sym_info:
                continue
            tick = mt5.symbol_info_tick(p.symbol)
            if not tick:
                continue

            side = "BUY" if p.type == mt5.ORDER_TYPE_BUY else "SELL"
            side_color = C_GREEN if p.profit >= 0 else C_RED
            current_price = tick.bid if p.type == mt5.ORDER_TYPE_BUY else tick.ask

            if p.price_open > 0:
                pct_return = (current_price - p.price_open) / p.price_open * 100 if side == "BUY" else (p.price_open - current_price) / p.price_open * 100
            else:
                pct_return = 0.0

            ind = current_indicators.get(p.symbol, {})
            atr = ind.get("atr", 0.01)
            profit_atr = abs(current_price - p.price_open) / atr if atr > 0 else 0
            status = "Trailing ativo" if profit_atr >= 1.5 else "Breakeven" if profit_atr >= 1.0 else "Aguardando"

            line = f"{p.symbol:<6} {side:<4} {p.volume:<6.2f} {p.price_open:<9.2f} {current_price:<9.2f} " \
                   f"{side_color}{p.profit:>+10.2f}{C_RESET} {pct_return:+6.1f}% {status:<18}"
            print(f"║ {line} ║")

    print(f"╠{'═' * 90}╣")

    # ========================
    # TOP 15 OPORTUNIDADES
    # ========================
    print(f"║ {C_YELLOW}📊 TOP 15 OPORTUNIDADES (com motivo de decisão){C_RESET}{' '*35}║")
    print(f"║ {'SYM':<6} {'DIR':<4} {'RSI':<5} {'ATR':<6} {'VWAP':<7} {'MACRO':<6} {'DECISÃO':<12} {'MOTIVO PRINCIPAL':<30} ║")
    print(f"║ {'─'*88} ║")

    with top15_lock:
        symbols = list(current_top15)

    for sym in symbols:
        ind = current_indicators.get(sym)
        if not ind:
            continue

        params = optimized_params.get(sym, {})
        rsi_low = params.get("rsi_low", 35)
        rsi_high = params.get("rsi_high", 70)
        rsi_ok = rsi_low <= ind["rsi"] <= rsi_high
        ema_ok = ind["ema_fast"] > ind["ema_slow"]

        tick = mt5.symbol_info_tick(sym)
        if not tick: 
            continue
        price = (tick.ask + tick.bid)/2
        df_vwap = utils.safe_copy_rates(sym, current_timeframe, 100)
        vwap = utils.get_intraday_vwap(df_vwap)
        vwap_ok_long = not vwap or price > vwap
        vwap_ok_short = not vwap or price < vwap
        macro_long = macro_trend_ok(sym, "BUY")
        macro_short = macro_trend_ok(sym, "SELL")

        # Verifica se já tem posição nesse símbolo e direção
        positions = mt5.positions_get() or []
        has_long = any(p.symbol == sym and p.type == mt5.ORDER_TYPE_BUY for p in positions)
        has_short = any(p.symbol == sym and p.type == mt5.ORDER_TYPE_SELL for p in positions)
        has_pos = has_long or has_short

        # === CALCULA DECISÃO POSSÍVEL PRIMEIRO ===
        potential_long = ema_ok and rsi_ok and vwap_ok_long and macro_long and config.TRADE_BOTH_DIRECTIONS
        potential_short = (not ema_ok) and rsi_ok and vwap_ok_short and macro_short and config.TRADE_BOTH_DIRECTIONS

        if potential_long and not has_long:
            decision = "🟢 COMPRA"
            intended_side = "BUY"
        elif potential_short and not has_short:
            decision = "🔴 VENDA"
            intended_side = "SELL"
        else:
            decision = "⏸️  BLOQ."
            intended_side = None

        # === AGORA VERIFICA SE DEVERIA TER ENTRADO MAS FALHOU ===
        should_have_position = (potential_long and not has_long) or (potential_short and not has_short)
        failed_to_enter = should_have_position and not (has_long or has_short)

        reason = ""
        if failed_to_enter:
            with failure_lock:
                fail_reason = last_failure_reason.get(sym, "Tentativa recente (ver log)")
            reason = f"FALHA NA ORDEM: {fail_reason}"
            decision = "⚠️ FALHA COMPRA" if potential_long else "⚠️ FALHA VENDA"
        elif decision == "🟢 COMPRA" or decision == "🔴 VENDA":
            reason = "Sinal completo"
        else:
            parts = []
            if has_pos:
                parts.append("já posicionado")
            if not ema_ok and potential_long:
                parts.append("EMA contra")
            if not rsi_ok:
                parts.append("RSI fora")
            if vwap and price <= vwap and potential_long:
                parts.append("abaixo VWAP")
            if vwap and price >= vwap and potential_short:
                parts.append("acima VWAP")
            if not macro_long and potential_long:
                parts.append("macro baixa")
            if not macro_short and potential_short:
                parts.append("macro alta")
            sector = config.SECTOR_MAP.get(sym, "UNKNOWN")
            if get_sector_counts().get(sector, 0) >= config.MAX_PER_SECTOR:
                parts.append("setor no limite")
            if mt5.positions_total() >= config.MAX_SYMBOLS:
                parts.append("máx posições")
            reason = "; ".join(parts) or "OK (aguardando)"

        dir_arrow = "↑" if ema_ok else "↓"
        decision_color = C_RED if "FALHA" in decision else C_GREEN if "COMPRA" in decision or "VENDA" in decision else C_DIM

        line = f"{sym:<6} {dir_arrow:<4} {ind['rsi']:<5.1f} {ind['atr']:<6.2f} " \
               f"{'↑' if vwap_ok_long else '↓' if vwap_ok_short else '-':<7} " \
               f"{'↑' if macro_long else '↓' if macro_short else '-':<6} " \
               f"{decision_color}{decision:<12}{C_RESET} {reason:<30}"
        print(f"║ {line} ║")

    print(f"╚{'═' * 90}╝")

# =========================
# MAIN
# =========================
def main():
    logger.info("🚀 XP3 PRO BOT INICIANDO - VERSÃO BIDIRECIONAL + PYRAMIDING")
    
    if not mt5.initialize():
        logger.critical("❌ MT5 não inicializou")
        return

    ind, top = build_portfolio_and_top15()
    with top15_lock:
        global current_indicators, current_top15
        current_indicators = ind
        current_top15 = top

    threading.Thread(target=fast_loop, daemon=True).start()

    # Atualização lenta do Top15
    while True:
        time.sleep(1800)
        ind, top = build_portfolio_and_top15()
        with top15_lock:
            current_indicators = ind
            current_top15 = top
        logger.info(f"TOP15 atualizado → {top[:10]}...")

if __name__ == "__main__":
    main()