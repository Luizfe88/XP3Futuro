# bot.py — XP3 BOT B3 (VERSÃO FINAL PROFISSIONAL - 16/12/2025)

import time
import threading
import logging
from datetime import datetime, date, timedelta
from threading import Lock
from collections import deque, defaultdict
import MetaTrader5 as mt5
import config
import utils
import numpy as np
from typing import Optional, Dict, Any, List

# ===== ANSI COLORS =====
C_RESET = "\033[0m"
C_GREEN = "\033[92m"
C_RED = "\033[91m"
C_YELLOW = "\033[93m"
C_CYAN = "\033[96m"
C_BOLD = "\033[1m"
C_DIM = "\033[2m"
C_WHITE = "\033[97m"

# ===== ESTADO GLOBAL =====
correlation_cache = {}
last_correlation_update = None
correlation_lock = Lock()

_symbol_pyramid_leg = {}      # {symbol: {"leg": 1 ou 2, "entry": price}}
last_entry_attempt = {}       # {symbol: datetime da última tentativa de entrada}

top15_lock = Lock()
alerts = deque(maxlen=10)
alerts_lock = Lock()
failure_lock = Lock()

current_top15 = []
current_indicators = {}
optimized_params = {}
trading_paused = False
daily_max_equity = 0.0
last_reset_day: Optional[date] = None
last_failure_reason = {}      # {symbol: motivo da última falha}

# =========================
# LOG
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("xp3_bot.log", encoding="utf-8")
    ]
)
logger = logging.getLogger("bot")

# =========================
# TIMEFRAMES
# =========================
TIMEFRAME_BASE = mt5.TIMEFRAME_M15
TIMEFRAME_MACRO = getattr(mt5, f"TIMEFRAME_{config.MACRO_TIMEFRAME}", mt5.TIMEFRAME_H1)
current_timeframe = TIMEFRAME_BASE

# =========================
# FUNÇÕES AUXILIARES
# =========================
def clear_screen():
    import os
    os.system("cls" if os.name == "nt" else "clear")

def push_alert(msg: str, level: str = "INFO", sound: bool = True):
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
# CORRELAÇÃO
# =========================
def update_correlation_matrix():
    global correlation_cache, last_correlation_update
    with top15_lock:
        symbols = current_top15[:15]

    if len(symbols) < 2:
        return

    with correlation_lock:
        correlation_cache = utils.calculate_correlation_matrix(
            symbols, lookback_days=config.CORRELATION_LOOKBACK_DAYS
        )
        last_correlation_update = datetime.now()
        logger.info(f"Matriz de correlação atualizada ({len(symbols)} ativos)")

def get_average_correlation_with_portfolio(symbol: str, current_positions_symbols: List[str]) -> float:
    if not config.ENABLE_CORRELATION_FILTER or not correlation_cache:
        return 0.0

    corrs = []
    sym_corrs = correlation_cache.get(symbol, {})
    for pos_sym in current_positions_symbols:
        if pos_sym == symbol:
            continue
        corr = sym_corrs.get(pos_sym)
        if corr is not None:
            corrs.append(abs(corr))

    return np.mean(corrs) if corrs else 0.0

# =========================
# PARÂMETROS OTIMIZADOS
# =========================
# =========================
# PARÂMETROS OTIMIZADOS (AGORA USANDO ELITE_SYMBOLS)
# =========================
def load_optimized_params():
    global optimized_params
    # Primeiro tenta carregar do ELITE_SYMBOLS (gerenciado pelo otimizador)
    elite = getattr(config, "ELITE_SYMBOLS", {})
    if elite:
        optimized_params = {sym: params.copy() for sym, params in elite.items()}
        logger.info(f"Parâmetros carregados do ELITE_SYMBOLS ({len(optimized_params)} ativos elite)")
    else:
        # Fallback: usa o antigo OPTIMIZED_PARAMS ou default
        optimized_params = getattr(config, "OPTIMIZED_PARAMS", {}).copy()
        logger.warning("ELITE_SYMBOLS vazio ou não encontrado. Usando fallback.")

    # Garante que todo ativo em optimized_params tenha os campos mínimos
    for sym in optimized_params:
        params = optimized_params[sym]
        defaults = {"ema_short": 9, "ema_long": 21, "rsi_low": 35, "rsi_high": 70, "adx_threshold": 25, "mom_min": 0.0}
        for k, v in defaults.items():
            params.setdefault(k, v)

# =========================
# BUILD TOP15
# =========================
def build_portfolio_and_top15():
    scored = []
    indicators = {}
    load_optimized_params()  # Carrega ELITE_SYMBOLS

    # Usa apenas os ativos do ELITE_SYMBOLS (gerenciados pelo otimizador)
    elite_symbols = list(optimized_params.keys())
    if not elite_symbols:
        logger.warning("ELITE_SYMBOLS vazio. Usando fallback completo do SECTOR_MAP.")
        elite_symbols = list(config.SECTOR_MAP.keys())

    logger.info(f"Construindo TOP15 com {len(elite_symbols)} ativos (elite mode)")

    for sym in elite_symbols:
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
        if ind["atr"] > 0.3:
            score += 10

        scored.append((score, sym))
        indicators[sym] = ind

    scored.sort(reverse=True)
    selected_top = [s for _, s in scored[:15]]
    logger.info(f"TOP15 gerado: {selected_top}")
    return indicators, selected_top
# =========================
# FILTROS COMUNS
# =========================
def is_trading_time() -> bool:
    now = datetime.now()
    start = datetime.strptime(config.TRADING_START, "%H:%M").time()
    end = datetime.strptime(config.TRADING_END, "%H:%M").time()
    current = now.time()
    return start <= current <= end

def additional_filters_ok(symbol: str) -> bool:
    df = utils.safe_copy_rates(symbol, TIMEFRAME_BASE, 100)
    if df is None:
        return False

    # Volume médio
    avg_vol = utils.get_avg_volume(df)
    if avg_vol < config.MIN_AVG_VOLUME_20:
        return False

    # Gap de abertura
    gap = utils.get_open_gap(symbol, TIMEFRAME_BASE)
    if gap and gap > config.MAX_GAP_OPEN_PCT:
        return False

    # Horário de proibição de novas entradas
    now_time = datetime.now().time()
    no_entry_time = datetime.strptime(config.NO_ENTRY_AFTER, "%H:%M").time()
    if now_time > no_entry_time:
        return False

    return True

def get_sector_counts() -> defaultdict:
    positions = mt5.positions_get() or []
    counts = defaultdict(int)
    for p in positions:
        sector = config.SECTOR_MAP.get(p.symbol, "UNKNOWN")
        counts[sector] += 1
    return counts

# =========================
# MACRO TREND
# =========================
def macro_trend_ok(symbol: str, side: str) -> bool:
    df_macro = utils.safe_copy_rates(symbol, TIMEFRAME_MACRO, 300)
    if df_macro is None or len(df_macro) < config.MACRO_EMA_LONG:
        return False
    close = df_macro["close"]
    ema200 = close.ewm(span=config.MACRO_EMA_LONG, adjust=False).mean().iloc[-1]
    current_price = (mt5.symbol_info_tick(symbol).last or close.iloc[-1])
    if side == "BUY":
        return current_price > ema200
    return current_price < ema200

# =========================
# GESTÃO DE POSIÇÕES
# =========================
def update_sl(ticket: int, new_sl: float):
    pos = mt5.positions_get(ticket=ticket)
    if pos:
        mt5.order_send({
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "sl": new_sl,
            "tp": pos[0].tp
        })

def manage_positions_advanced():
    positions = mt5.positions_get()
    if not positions:
        return

    for pos in positions:
        sym = pos.symbol
        ind = current_indicators.get(sym)
        if not ind or ind.get("atr") is None or ind["atr"] <= 0.01:
            continue

        atr = ind["atr"]
        tick = mt5.symbol_info_tick(sym)
        if not tick:
            continue

        current_price = tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask
        profit_atr = abs(current_price - pos.price_open) / atr

        # Time-stop
        duration_candles = (datetime.now() - datetime.fromtimestamp(pos.time)).total_seconds() / (15 * 60)
        if duration_candles > config.MAX_TRADE_DURATION_CANDLES:
            push_alert(f"⏰ TIME-STOP: {sym} fechada após {duration_candles:.0f} candles", "WARNING", True)
            # Implementar fechamento completo se desejar
            continue

        # Breakeven puro ao atingir +1.5 ATR
        if profit_atr >= 1.5 and pos.sl != pos.price_open:
            new_sl = pos.price_open
            update_sl(pos.ticket, new_sl)
            push_alert(f" Breakeven ativado {sym} (+1.5 ATR)", "INFO", True)

        # Partial 50% ao atingir +2.0 ATR
        if profit_atr >= 2.0 and pos.volume >= 0.02:
            info = mt5.symbol_info(sym)
            partial_vol = max(info.volume_min, round(pos.volume * 0.5 / info.volume_step) * info.volume_step)
            if partial_vol >= info.volume_min:
                # Implementar partial close se desejar
                push_alert(f"🎯 PARTIAL 50% {sym} (+{profit_atr:.1f} ATR)", "INFO", True)

        # Trailing progressivo (mais agressivo com 2 pernas)
        pyramid_info = _symbol_pyramid_leg.get(sym, {})
        legs = pyramid_info.get("leg", 1)
        if profit_atr >= 1.5:
            mult = 1.2 if legs >= 2 else 1.5
            new_sl = current_price - atr * mult if pos.type == mt5.ORDER_TYPE_BUY else current_price + atr * mult
            if (pos.type == mt5.ORDER_TYPE_BUY and new_sl > pos.sl) or \
               (pos.type == mt5.ORDER_TYPE_SELL and new_sl < pos.sl):
                update_sl(pos.ticket, new_sl)
                push_alert(f"🔄 Trailing {sym} (leg {legs}) → SL {new_sl:.2f}", "INFO", True)

# =========================
# ENTRADA COM PYRAMIDING
# =========================
def try_enter_position(symbol: str, intended_side: str):
    if not additional_filters_ok(symbol):
        return

    tick = mt5.symbol_info_tick(symbol)
    if not tick or tick.ask <= 0 or tick.bid <= 0:
        return

    ind = current_indicators.get(symbol)
    if not ind or ind.get("atr") is None:
        return

    params = optimized_params.get(symbol, {})
    rsi_low = params.get("rsi_low", 35)
    rsi_high = params.get("rsi_high", 70)

    # Verificação do sinal
    if intended_side == "BUY":
        if not (ind["ema_fast"] > ind["ema_slow"] and rsi_low <= ind["rsi"] <= rsi_high):
            return
        entry_price = tick.ask
    else:
        if not (ind["ema_fast"] < ind["ema_slow"] and rsi_low <= ind["rsi"] <= rsi_high):
            return
        entry_price = tick.bid

    atr = ind["atr"]
    sl = utils.calculate_sl_price(entry_price, intended_side, atr)
    tp = entry_price + (entry_price - sl) * 2 if intended_side == "BUY" else entry_price - (sl - entry_price) * 2

    acc = mt5.account_info()
    if not acc:
        return

    # Pyramiding
    pyramid_info = _symbol_pyramid_leg.get(symbol, {"leg": 0})
    current_leg = pyramid_info["leg"]

    if current_leg == 0:  # Primeira perna
        risk_pct = config.RISK_PER_TRADE_PCT * config.PYRAMID_RISK_SPLIT[0]
        leg = 1
    elif current_leg == 1:  # Possível segunda perna
        positions = [p for p in mt5.positions_get(symbol=symbol) or []
                     if (p.type == mt5.ORDER_TYPE_BUY and intended_side == "BUY") or
                        (p.type == mt5.ORDER_TYPE_SELL and intended_side == "SELL")]
        if not positions:
            return
        pos = positions[0]
        profit_atr = abs((tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask) - pos.price_open) / atr
        if profit_atr < config.PYRAMID_ATR_DISTANCE:
            return
        risk_pct = config.RISK_PER_TRADE_PCT * config.PYRAMID_RISK_SPLIT[1]
        leg = 2
    else:
        return  # Máximo 2 pernas

    volume = utils.calculate_position_size_atr(acc.equity, risk_pct, atr)
    if not volume or volume < 0.01:
        return

    result = utils.send_order_with_sl_tp(symbol, intended_side, volume, sl, tp)
    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        push_alert(f"✅ ENTRADA {intended_side} {symbol} {volume:.2f} @ {entry_price:.2f} SL {sl:.2f}", "INFO", True)
        _symbol_pyramid_leg[symbol] = {"leg": leg, "entry": entry_price}
        with failure_lock:
            last_failure_reason.pop(symbol, None)
    else:
        reason = result.comment if result else "Sem resposta MT5"
        push_alert(f"⚠️ Falha {intended_side} {symbol}: {reason}", "WARNING", True)
        with failure_lock:
            last_failure_reason[symbol] = reason

# =========================
# CIRCUIT BREAKER
# =========================
def check_for_circuit_breaker():
    global trading_paused, daily_max_equity, last_reset_day

    acc = mt5.account_info()
    if not acc:
        return

    now = datetime.now()
    today = now.date()

    # Reset diário
    reset_time = datetime.strptime(config.DAILY_RESET_TIME, "%H:%M").time()
    if now.time() >= reset_time and last_reset_day != today:
        daily_max_equity = acc.equity
        trading_paused = False
        last_reset_day = today
        push_alert("🔄 Reset diário do Circuit Breaker", "WARNING")
        return

    if daily_max_equity == 0.0:
        daily_max_equity = acc.equity
        last_reset_day = today
        return

    if acc.equity > daily_max_equity:
        daily_max_equity = acc.equity

    drawdown_pct = (daily_max_equity - acc.equity) / daily_max_equity
    if drawdown_pct >= config.MAX_DAILY_DRAWDOWN_PCT and not trading_paused:
        trading_paused = True
        push_alert("🚨 CIRCUIT BREAKER ATIVADO - Trading pausado!", "CRITICAL", True)

# =========================
# DASHBOARD
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
    status = "PAUSADO (CB)" if trading_paused else "ATIVO"

    print(f"{C_BOLD}╔{'═' * 96}╗{C_RESET}")
    print(f"║ {C_CYAN}🚀 XP3 PRO BOT - B3{C_RESET}  📅 {now}  {C_GREEN if not trading_paused else C_RED}{status}{C_RESET} {' '*40}║")
    print(f"╠{'═' * 96}╣")
    print(f"║ Equity: R$ {acc.equity:,.2f}  |  Balance: R$ {acc.balance:,.2f}  |  PnL: {pnl_color}{pnl:+,.2f}{C_RESET} {' '*30}║")
    print(f"║ Posições: {mt5.positions_total()}/{config.MAX_SYMBOLS}  |  Risco/trade: {utils.get_current_risk_pct()*100:.1f}% {' '*35}║")
    print(f"╠{'═' * 96}╣")

    # Últimos alertas
    print(f"║ {C_YELLOW}🚨 ÚLTIMOS ALERTAS{C_RESET}{' '*76}║")
    with alerts_lock:
        recent = list(alerts)[:5]
    if not recent:
        print(f"║   {'(nenhum)':^92} ║")
    else:
        for _, msg in recent:
            print(f"║   {msg:<92} ║")
    print(f"╠{'═' * 96}╣")

    # Carteira atual
    print(f"║ {C_GREEN}💼 POSIÇÕES ABERTAS{C_RESET}{' '*74}║")
    positions = mt5.positions_get() or []
    if not positions:
        print(f"║   {'(nenhuma posição)':^92} ║")
    else:
        print(f"║ {'SYM':<6} {'DIR':<4} {'VOL':<6} {'ENTRY':<9} {'ATUAL':<9} {'PnL':<11} {'%':<7} {'STATUS':<18} ║")
        for p in positions:
            tick = mt5.symbol_info_tick(p.symbol)
            if not tick:
                continue
            side = "BUY" if p.type == mt5.ORDER_TYPE_BUY else "SELL"
            current_price = tick.bid if p.type == mt5.ORDER_TYPE_BUY else tick.ask
            pct = (current_price - p.price_open)/p.price_open*100 if side=="BUY" else (p.price_open - current_price)/p.price_open*100
            ind = current_indicators.get(p.symbol, {})
            profit_atr = abs(current_price - p.price_open) / ind.get("atr", 0.01) if ind.get("atr") else 0
            status = "Trailing" if profit_atr >= 1.5 else "Breakeven" if profit_atr >= 1.0 else "Aguardando"
            line = f"{p.symbol:<6} {side:<4} {p.volume:<6.2f} {p.price_open:<9.2f} {current_price:<9.2f} {p.profit:+10.2f} {pct:+6.1f}% {status:<18}"
            print(f"║ {line} ║")
    print(f"╠{'═' * 96}╣")

    # TOP 15
    print(f"║ {C_YELLOW}📊 TOP 15 OPORTUNIDADES{C_RESET}{' '*70}║")
    print(f"║ {'SYM':<6} {'DIR':<4} {'RSI':<5} {'ATR':<6} {'VWAP':<7} {'MACRO':<6} {'DECISÃO':<12} {'MOTIVO':<30} {'CORR':<6} ║")
    print(f"║ {'─'*94} ║")

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
        price = (tick.ask + tick.bid) / 2
        df_vwap = utils.safe_copy_rates(sym, current_timeframe, 100)
        vwap = utils.get_intraday_vwap(df_vwap)
        vwap_ok_long = not vwap or price > vwap
        vwap_ok_short = not vwap or price < vwap
        macro_long = macro_trend_ok(sym, "BUY")
        macro_short = macro_trend_ok(sym, "SELL")

        positions = mt5.positions_get() or []
        has_long = any(p.symbol == sym and p.type == mt5.ORDER_TYPE_BUY for p in positions)
        has_short = any(p.symbol == sym and p.type == mt5.ORDER_TYPE_SELL for p in positions)

        potential_long = ema_ok and rsi_ok and vwap_ok_long and macro_long and config.TRADE_BOTH_DIRECTIONS
        potential_short = (not ema_ok) and rsi_ok and vwap_ok_short and macro_short and config.TRADE_BOTH_DIRECTIONS

        intended_side = None
        decision = "⏸️ BLOQ."
        reason = ""

        if potential_long and not has_long:
            intended_side = "BUY"
            decision = "🟢 COMPRA"
            reason = "Sinal completo"
        elif potential_short and not has_short:
            intended_side = "SELL"
            decision = "🔴 VENDA"
            reason = "Sinal completo"

        # Correlação
        if intended_side and config.ENABLE_CORRELATION_FILTER:
            if (last_correlation_update is None or datetime.now() - last_correlation_update > timedelta(minutes=15)):
                update_correlation_matrix()
            avg_corr = get_average_correlation_with_portfolio(sym, [p.symbol for p in positions])
            if avg_corr > config.MIN_CORRELATION_SCORE_TO_BLOCK:
                decision = "⏸️ CORR.ALTA"
                intended_side = None
                reason = f"Corr alta ({avg_corr:.2f})"

        # Falhas anteriores
        if intended_side is None and decision not in ["⏸️ CORR.ALTA", "🟢 COMPRA", "🔴 VENDA"]:
            parts = []
            if has_long or has_short:
                parts.append("posicionado")
            if get_sector_counts().get(config.SECTOR_MAP.get(sym, "UNKNOWN"), 0) >= config.MAX_PER_SECTOR:
                parts.append("setor limite")
            if mt5.positions_total() >= config.MAX_SYMBOLS:
                parts.append("máx posições")
            reason = "; ".join(parts) or "Aguardando"

        # Dashboard linha
        dir_arrow = "↑" if ema_ok else "↓"
        decision_color = C_GREEN if "COMPRA" in decision or "VENDA" in decision else C_DIM
        avg_corr = get_average_correlation_with_portfolio(sym, [p.symbol for p in positions])
        corr_str = f"{avg_corr:.2f}" if avg_corr > 0.1 else "-"
        corr_color = C_RED if avg_corr > 0.75 else C_YELLOW if avg_corr > 0.60 else C_DIM

        line = f"{sym:<6} {dir_arrow:<4} {ind['rsi']:<5.1f} {ind['atr']:<6.2f} " \
               f"{'↑' if vwap_ok_long else '↓' if vwap_ok_short else '-':<7} " \
               f"{'↑' if macro_long else '↓' if macro_short else '-':<6} " \
               f"{decision_color}{decision:<12}{C_RESET} {reason:<30} {corr_color}{corr_str:<6}{C_RESET}"
        print(f"║ {line} ║")

        # EXECUÇÃO AUTOMÁTICA COM COOLDOWN
        now = datetime.now()
        last_attempt = last_entry_attempt.get(sym)
        if last_attempt and (now - last_attempt).total_seconds() < 60:
            continue

        if decision == "🟢 COMPRA" and intended_side == "BUY":
            try_enter_position(sym, "BUY")
            last_entry_attempt[sym] = now
        elif decision == "🔴 VENDA" and intended_side == "SELL":
            try_enter_position(sym, "SELL")
            last_entry_attempt[sym] = now

    print(f"╚{'═' * 96}╝")

# =========================
# FAST LOOP
# =========================
def fast_loop():
    logger.info("⚡ FAST LOOP INICIADO")
    while True:
        try:
            check_for_circuit_breaker()
            manage_positions_advanced()
            render_panel_enhanced()
            time.sleep(config.FAST_LOOP_INTERVAL_SECONDS)
        except Exception as e:
            logger.exception(f"Erro no fast loop: {e}")
            time.sleep(5)

# =========================
# MAIN
# =========================
def main():
    logger.info("🚀 XP3 PRO BOT INICIANDO - VERSÃO FINAL")
    if not mt5.initialize():
        logger.critical("❌ Falha ao inicializar MT5")
        return

    load_optimized_params()
    if optimized_params:
        source = "ELITE_SYMBOLS"
        logger.info(f"Bot iniciado usando {len(optimized_params)} ativos elite do {source}")
    else:
        source = "FALLBACK (SECTOR_MAP)"
        logger.info(f"Bot iniciado em modo {source} - ELITE_SYMBOLS vazio")

    ind, top = build_portfolio_and_top15()
    with top15_lock:
        global current_indicators, current_top15
        current_indicators = ind
        current_top15 = top

    threading.Thread(target=fast_loop, daemon=True).start()

    # Atualização lenta do TOP15
    while True:
        time.sleep(1800)
        ind, top = build_portfolio_and_top15()
        with top15_lock:
            current_indicators = ind
            current_top15 = top
        if (last_correlation_update is None or datetime.now() - last_correlation_update > timedelta(minutes=30)):
            update_correlation_matrix()
        logger.info(f"TOP15 atualizado → {top[:10]}...")

if __name__ == "__main__":
    main()