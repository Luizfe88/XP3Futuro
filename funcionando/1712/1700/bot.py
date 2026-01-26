# bot.py — XP3 BOT B3 (VERSÃO FINAL PROFISSIONAL - 17/12/2025)

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
from utils import send_telegram_trade, check_and_close_orphans, calculate_signal_score, safe_copy_rates

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
equity_inicio_dia = 0.0      
last_reset_day: Optional[date] = None
last_failure_reason = {}  
elite = getattr(config, "ELITE_SYMBOLS", {})
check_and_close_orphans(elite)    

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
def load_optimized_params():
    global optimized_params
    elite = getattr(config, "ELITE_SYMBOLS", {})
    if elite:
        optimized_params = {sym: params.copy() for sym, params in elite.items()}
        logger.info(f"Parâmetros carregados do ELITE_SYMBOLS ({len(optimized_params)} ativos elite)")
    else:
        optimized_params = getattr(config, "OPTIMIZED_PARAMS", {}).copy()
        logger.warning("ELITE_SYMBOLS vazio. Usando fallback.")

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
    load_optimized_params()

    elite_symbols = list(optimized_params.keys())
    if not elite_symbols:
        logger.warning("ELITE_SYMBOLS vazio. Usando fallback do SECTOR_MAP.")
        elite_symbols = list(config.SECTOR_MAP.keys())

    logger.info(f"Construindo TOP15 com {len(elite_symbols)} ativos")

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

    avg_vol = utils.get_avg_volume(df)
    if avg_vol < config.MIN_AVG_VOLUME_20:
        return False

    gap = utils.get_open_gap(symbol, TIMEFRAME_BASE)
    if gap and gap > config.MAX_GAP_OPEN_PCT:
        return False

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
def modify_sl(symbol: str, ticket: int, new_sl: float):
    pos = mt5.positions_get(ticket=ticket)
    if not pos:
        return
    pos = pos[0]

    request = {
        "action": mt5.TRADE_ACTION_SLTP,
        "position": ticket,
        "symbol": symbol,
        "sl": new_sl,
        "tp": pos.tp
    }
    mt5.order_send(request)

def close_position(symbol: str, ticket: int, volume: float, price: float, reason: str = ""):
    pos = mt5.positions_get(ticket=ticket)
    if not pos:
        return
    pos = pos[0]
    side = "BUY" if pos.type == mt5.ORDER_TYPE_BUY else "SELL"
    entry = pos.price_open

    profit_loss_points = (price - entry) if side == "BUY" else (entry - price)
    profit_loss_money = profit_loss_points * volume
    pl_pct = (profit_loss_money / (entry * volume)) * 100 if volume > 0 else 0

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": mt5.ORDER_TYPE_SELL if side == "BUY" else mt5.ORDER_TYPE_BUY,
        "position": ticket,
        "price": price,
        "deviation": 20,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
        "comment": f"XP3 Close - {reason}"
    }

    result = mt5.order_send(request)

    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        pl_color = "🟢" if profit_loss_money > 0 else "🔴"
        push_alert(f"{pl_color} FECHADO {side} {symbol} | {volume:.0f} ações | P&L: R${profit_loss_money:+.2f} ({pl_pct:+.2f}%) | {reason}")

        try:
            from utils import send_telegram_exit
            send_telegram_exit(
                symbol=symbol,
                side=side,
                volume=volume,
                entry_price=entry,
                exit_price=price,
                profit_loss=profit_loss_money
            )
        except Exception as e:
            logger.warning(f"Erro Telegram saída: {e}")

    else:
        push_alert(f"❌ Falha ao fechar {symbol}: {result.comment if result else 'Desconhecido'}", "WARNING")

def manage_positions_advanced():
    positions = mt5.positions_get()
    if not positions:
        return

    current_time = datetime.now()

    for pos in positions:
        symbol = pos.symbol
        ticket = pos.ticket
        side = "BUY" if pos.type == mt5.ORDER_TYPE_BUY else "SELL"
        volume_total = pos.volume
        entry_price = pos.price_open
        current_price = pos.price_current

        ind = current_indicators.get(symbol, {})
        atr = ind.get("atr")
        if not atr or atr <= 0:
            continue

        profit_points = (current_price - entry_price) if side == "BUY" else (entry_price - current_price)
        profit_atr = profit_points / atr

        df = utils.safe_copy_rates(symbol, TIMEFRAME_BASE, 100)
        if df is None or len(df) < 2:
            continue
        entry_time_approx = df.index[-1] - timedelta(minutes=15 * (len(df) - 1))
        candles_since_entry = max(1, int((current_time - entry_time_approx).total_seconds() / (15 * 60)))

        if candles_since_entry >= config.MAX_TRADE_DURATION_CANDLES:
            close_position(symbol, ticket, volume_total, current_price, reason="Time-stop")
            continue

        if config.ENABLE_BREAKEVEN and profit_atr >= config.BREAKEVEN_ATR_MULT and pos.sl <= entry_price + 0.01:
            new_sl = entry_price + 0.01 if side == "BUY" else entry_price - 0.01
            modify_sl(symbol, ticket, new_sl)
            push_alert(f"🔒 Breakeven ativado {symbol} @ R${new_sl:.2f}")

        if config.ENABLE_PARTIAL_CLOSE and profit_atr >= config.PARTIAL_CLOSE_ATR_MULT:
            partial_volume = round(volume_total * config.PARTIAL_PERCENT / 0.01) * 0.01
            if partial_volume >= 100:
                close_position(symbol, ticket, partial_volume, current_price, reason="Partial profit")
                push_alert(f"💰 Partial close {partial_volume} {symbol} em +{profit_atr:.2f} ATR")

        if config.ENABLE_TRAILING_STOP:
            trail_mult = config.TRAILING_ATR_MULT_TIGHT if profit_atr >= 3.0 else config.TRAILING_ATR_MULT_INITIAL
            new_sl = current_price - (atr * trail_mult) if side == "BUY" else current_price + (atr * trail_mult)
            if (side == "BUY" and new_sl > pos.sl) or (side == "SELL" and new_sl < pos.sl):
                modify_sl(symbol, ticket, new_sl)
                tight = " (apertado)" if profit_atr >= 3.0 else ""
                push_alert(f"📉 Trailing stop atualizado{tight} {symbol} → R${new_sl:.2f}")

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

    pyramid_info = _symbol_pyramid_leg.get(symbol, {"leg": 0})
    current_leg = pyramid_info["leg"]

    # Define risco e perna com base na leg atual
    if current_leg == 0:
        risk_pct = config.RISK_PER_TRADE_PCT * config.PYRAMID_RISK_SPLIT[0]
        leg = 1
    elif current_leg == 1:
        # Verifica se já tem posição na direção correta
        positions = [p for p in mt5.positions_get(symbol=symbol) or []
                     if (p.type == mt5.ORDER_TYPE_BUY and intended_side == "BUY") or
                        (p.type == mt5.ORDER_TYPE_SELL and intended_side == "SELL")]
        if not positions:
            return
        pos = positions[0]
        profit_atr = abs((tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask) - pos.price_open) / atr
        if profit_atr < config.PYRAMID_ATR_DISTANCE:
            return

        # <<< CHECK DE CORRELAÇÃO ANTES DA SEGUNDA PERNA >>>
        if config.ENABLE_CORRELATION_FILTER:
            update_correlation_matrix()
            current_positions = mt5.positions_get() or []
            current_symbols = [p.symbol for p in current_positions]
            avg_corr = get_average_correlation_with_portfolio(symbol, current_symbols)
            if avg_corr > 0.70:
                push_alert(f"⚠️ Pyramiding bloqueado {symbol}: Correlação alta ({avg_corr:.2f})")
                return

        risk_pct = config.RISK_PER_TRADE_PCT * config.PYRAMID_RISK_SPLIT[1]
        leg = 2
    else:
        return  # Máximo 2 pernas

    # Agora risk_pct e leg estão sempre definidos
    volume = utils.calculate_position_size_atr(acc.equity, risk_pct, atr)
    if not volume or volume < 0.01:
        return

    result = utils.send_order_with_sl_tp(symbol, intended_side, volume, sl, tp)
    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        push_alert(f"✅ ENTRADA {intended_side} {symbol} {volume:.2f} @ {entry_price:.2f} SL {sl:.2f}", "INFO", True)
        _symbol_pyramid_leg[symbol] = {"leg": leg, "entry": entry_price}
        with failure_lock:
            last_failure_reason.pop(symbol, None)

        try:
            send_telegram_trade(
                symbol=symbol,
                side=intended_side,
                volume=volume,
                price=entry_price,
                sl=sl,
                tp=tp
            )
        except Exception as e:
            logger.warning(f"Falha Telegram entrada: {e}")
    else:
        reason = result.comment if result else "Sem resposta MT5"
        push_alert(f"⚠️ Falha {intended_side} {symbol}: {reason}", "WARNING", True)
        with failure_lock:
            last_failure_reason[symbol] = reason

# =========================
# CIRCUIT BREAKER
# =========================
def check_for_circuit_breaker():
    global trading_paused, daily_max_equity, equity_inicio_dia, last_reset_day

    acc = mt5.account_info()
    if not acc:
        return

    now = datetime.now()
    today = now.date()

    reset_time = datetime.strptime(config.DAILY_RESET_TIME, "%H:%M").time()
    if now.time() >= reset_time and last_reset_day != today:
        daily_max_equity = acc.equity
        equity_inicio_dia = acc.equity
        trading_paused = False
        last_reset_day = today
        push_alert("🔄 Reset diário do Circuit Breaker", "WARNING")
        return

    if daily_max_equity == 0.0:
        daily_max_equity = acc.equity
        equity_inicio_dia = acc.equity
        last_reset_day = today
        return

    if acc.equity > daily_max_equity:
        daily_max_equity = acc.equity

    drawdown_pct = (daily_max_equity - acc.equity) / daily_max_equity
    if drawdown_pct >= config.MAX_DAILY_DRAWDOWN_PCT and not trading_paused:
        trading_paused = True
        push_alert("🚨 CIRCUIT BREAKER ATIVADO - Trading pausado!", "CRITICAL", True)

# =========================
# RELATÓRIO DIÁRIO
# =========================
def daily_report():
    while True:
        now = datetime.now()
        if now.hour == 18 and now.minute < 5:
            acc = mt5.account_info()
            if acc and equity_inicio_dia > 0:
                pnl_day = acc.equity - equity_inicio_dia
                msg = f"📊 <b>RELATÓRIO DIÁRIO XP3 - {now.strftime('%d/%m/%Y')}</b>\n\n"
                msg += f"Equity: R${acc.equity:,.2f}\n"
                msg += f"PnL do dia: <b>{pnl_day:+.2f}</b>\n"
                msg += f"Posições abertas: {mt5.positions_total()}\n"
                msg += f"Status: {'PAUSADO' if trading_paused else 'ATIVO'}"

                try:
                    bot = utils.get_telegram_bot()
                    bot.send_message(config.TELEGRAM_CHAT_ID, msg, parse_mode="HTML")
                    print("✅ Relatório diário enviado!")
                except Exception as e:
                    logger.warning(f"Erro ao enviar relatório: {e}")
            time.sleep(300)
        time.sleep(60)

# =========================
# DASHBOARD (mantido igual ao seu - excelente!)
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

    # =========================
    # POSIÇÕES ABERTAS
    # =========================
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

        # =========================
    # TOP 15 ATIVOS DO ELITE + CANDLE E PREÇO
    # =========================
    print(f"║ {C_YELLOW}📊 TOP 15 ATIVOS DO ELITE + CANDLE E PREÇO{C_RESET}{' '*50}║")
    print(f"║ {'RANK':<4} {'SYM':<6} {'SCORE':<6} {'DIR':<4} {'RSI':<5} {'ATR%':<6} {'CANDLE':<7} {'PREÇO':<8} {'CORR':<5} {'SETOR':<12} {'STATUS':<15} {'MOTIVO':<20} ║")

    with top15_lock:
        top15_symbols = list(current_top15)

    if not top15_symbols:
        print(f"║   {'TOP15 vazio ou não carregado':^92} ║")
    else:
        positions = mt5.positions_get() or []
        current_symbols = [p.symbol for p in positions]
        current_sectors = get_sector_counts()

        for rank, sym in enumerate(top15_symbols, 1):
            ind = current_indicators.get(sym)
            if not ind:
                print(f"║ {rank:<4} {sym:<6} {'0.0':<6} {'-':<4} {'-':<5} {'-':<6} {'N/A':<7} {'-':<8} {'-':<5} {'UNKNOWN':<12} {'⏸️ ERRO':<15} {'Sem indicadores':<20} ║")
                continue

            params = optimized_params.get(sym, {})
            rsi_low = params.get("rsi_low", 35)
            rsi_high = params.get("rsi_high", 70)
            adx_threshold = params.get("adx_threshold", 25)

            # Sinal técnico
            long_signal = (ind["ema_fast"] > ind["ema_slow"] and 
                           rsi_low <= ind["rsi"] <= rsi_high and 
                           ind.get("adx", 0) >= adx_threshold)
            short_signal = (ind["ema_fast"] < ind["ema_slow"] and 
                            rsi_low <= ind["rsi"] <= rsi_high and 
                            ind.get("adx", 0) >= adx_threshold)

            dir_arrow = "↑" if long_signal else "↓" if short_signal else "-"
            side = "BUY" if long_signal else "SELL" if short_signal else None

            # Último candle e preço
            df_last = utils.safe_copy_rates(sym, TIMEFRAME_BASE, 1)
            if df_last is not None and not df_last.empty:
                last_candle_time = df_last.index[-1].strftime("%H:%M")
                last_close = df_last["close"].iloc[-1]
                price_str = f"{last_close:.2f}"
            else:
                last_candle_time = "N/A"
                price_str = "-"

            if not (long_signal or short_signal):
                motive = "Sem sinal técnico"
                status = "⏸️ BLOQ."
                score = 0.0
            else:
                if not additional_filters_ok(sym):
                    motive = "Filtro (vol/gap/horário)"
                    status = "⏸️ FILTRO"
                    score = utils.calculate_signal_score(sym, ind, params, current_symbols, TIMEFRAME_BASE)
                elif not macro_trend_ok(sym, side):
                    motive = "Contra macro trend"
                    status = "⏸️ MACRO"
                    score = utils.calculate_signal_score(sym, ind, params, current_symbols, TIMEFRAME_BASE)
                elif any(p.symbol == sym for p in positions):
                    motive = "Já posicionado"
                    status = "✔️ ABERTO"
                    score = 0.0
                elif current_sectors[config.SECTOR_MAP.get(sym, "UNKNOWN")] >= config.MAX_PER_SECTOR:
                    motive = "Limite por setor"
                    status = "⏸️ SETOR"
                    score = utils.calculate_signal_score(sym, ind, params, current_symbols, TIMEFRAME_BASE)
                elif len(positions) >= config.MAX_SYMBOLS:
                    motive = "Carteira cheia"
                    status = "⏸️ CHEIO"
                    score = utils.calculate_signal_score(sym, ind, params, current_symbols, TIMEFRAME_BASE)
                else:
                    score = utils.calculate_signal_score(sym, ind, params, current_symbols, TIMEFRAME_BASE)
                    if score < 40:
                        motive = "Score baixo"
                        status = "⏸️ SCORE"
                    else:
                        motive = "Sinal forte"
                        status = "🟢 ENTRANDO" if len(positions) < config.MAX_SYMBOLS else "⏸️ CHEIO"
                        now = datetime.now()
                        last_attempt = last_entry_attempt.get(sym)
                        if not last_attempt or (now - last_attempt).total_seconds() > 90:
                            try_enter_position(sym, side)
                            last_entry_attempt[sym] = now

            # Métricas para exibição
            close_price = ind.get("close", 0) or 0.01
            atr_pct = (ind["atr"] / close_price) * 100 if close_price > 0 else 0
            avg_corr = get_average_correlation_with_portfolio(sym, current_symbols) if config.ENABLE_CORRELATION_FILTER else 0
            corr_str = f"{avg_corr:.2f}" if avg_corr > 0 else "-"
            corr_color = C_RED if avg_corr > 0.7 else C_YELLOW if avg_corr > 0.5 else C_GREEN

            sector = config.SECTOR_MAP.get(sym, "UNKNOWN")[:11]

            status_color = C_GREEN if "ENTRANDO" in status or "ABERTO" in status else C_YELLOW if "AGUARD" in status else C_RED

            motive_display = motive[:19] + "..." if len(motive) > 19 else motive

            line = f"{rank:<4} {sym:<6} {score:<6.1f} {dir_arrow:<4} {ind['rsi']:<5.1f} {atr_pct:<6.2f} " \
                   f"{last_candle_time:<7} {price_str:<8} {corr_color}{corr_str:<5}{C_RESET} {sector:<12} " \
                   f"{status_color}{status:<15}{C_RESET} {motive_display:<20}"
            print(f"║ {line} ║")

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
    source = "ELITE_SYMBOLS" if optimized_params else "FALLBACK (SECTOR_MAP)"
    logger.info(f"Bot iniciado usando {len(optimized_params)} ativos do {source}")

    ind, top = build_portfolio_and_top15()
    with top15_lock:
        global current_indicators, current_top15
        current_indicators = ind
        current_top15 = top

    threading.Thread(target=fast_loop, daemon=True).start()
    threading.Thread(target=daily_report, daemon=True).start()  # <<< CORRETO

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