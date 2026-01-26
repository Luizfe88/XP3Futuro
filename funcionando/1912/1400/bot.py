# bot.py — XP3 BOT B3 (VERSÃO FINAL PROFISSIONAL - 18/12/2025)
import pandas as pd
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
from utils import (
    send_telegram_trade, send_telegram_exit, get_telegram_bot,
    check_and_close_orphans, calculate_signal_score, safe_copy_rates,
    detect_market_regime, calculate_sector_exposure_pct, macro_trend_ok
)
from ml_optimizer import ml_optimizer
import os
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
_symbol_pyramid_leg = {}
last_entry_attempt = {}
top15_lock = Lock()
alerts = deque(maxlen=10)
alerts_lock = Lock()
failure_lock = Lock()
current_top15 = []
current_indicators_lock = Lock()
optimized_params = {}
trading_paused = False
daily_max_equity = 0.0
equity_inicio_dia = 0.0      
last_reset_day: Optional[date] = None
last_failure_reason = {}  
_last_eod_report_date = None
entry_indicators = {}  # {symbol: indicadores_da_entrada}
entry_indicators_lock = Lock()
# Flag para controle de mensagens de inicialização
_first_build_done = False
_first_build_lock = Lock()

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

def push_panel_alert(msg: str, sound: bool = False):
    """
    Adiciona mensagem apenas ao painel (últimos alertas)
    NÃO grava no log nem imprime no console
    """
    timestamp = datetime.now().strftime("%H:%M:%S")
    entry = f"[{timestamp}] {msg}"

    with alerts_lock:
        alerts.appendleft(("INFO", entry))

    if sound:
        try:
            print("\a", end="")
        except Exception:
            pass
# =========================
# THREAD DE CORRELAÇÃO (fora do main)
# =========================
def correlation_updater_thread():
    while True:
        try:
            time.sleep(1800)
            update_correlation_matrix()
        except Exception as e:
            logger.error(f"Erro na thread de correlação: {e}", exc_info=True)
            time.sleep(300)  # Aguarda 5min e tenta novamente

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
            symbols
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
    global _first_build_done
    scored = []
    indicators = {}
    load_optimized_params()

    elite_symbols = list(optimized_params.keys())
    
    if not elite_symbols:
        logger.error("❌ ELITE_SYMBOLS está vazio!")
        return {}, []

    # Mensagem de carregamento apenas na primeira vez
    with _first_build_lock:
        if not _first_build_done:
            logger.info(f"Parâmetros carregados do ELITE_SYMBOLS ({len(elite_symbols)} ativos elite)")
            _first_build_done = True

    for sym in elite_symbols:
        df = utils.safe_copy_rates(sym, TIMEFRAME_BASE, 300)
        
        if df is None:
            mt5.symbol_select(sym, True)
            time.sleep(0.5)
            df = utils.safe_copy_rates(sym, TIMEFRAME_BASE, 300)
            
        if df is None or len(df) < 20:
            ind = {
                "ema_fast": 0, "ema_slow": 0, "rsi": 50, "atr": 0.01,
                "atr_pct": 0, "adx": 0, "vwap": None, "close": 0,
                "macro_trend_ok": False, "tick_size": 0.01,
                "sector": config.SECTOR_MAP.get(sym, "Elite"),
                "error": "NO_DATA"
            }
            scored.append((1, sym))
            indicators[sym] = ind
            # Mensagem apenas na primeira carga
            with _first_build_lock:
                if not _first_build_done:  # Já foi setado acima, mas por segurança
                    logger.info(f"⚠️ {sym}: Sem dados, mantido no TOP15 com score 1")
            continue

        params = optimized_params.get(sym, {})
        ind = utils.get_cached_indicators(sym, TIMEFRAME_BASE, 300)
        if ind.get("error"):
            # fallback manual se necessário
            df = utils.safe_copy_rates(sym, TIMEFRAME_BASE, 300)
            ind = utils.quick_indicators_custom(sym, TIMEFRAME_BASE, df=df, params=params) if df is not None else {"error": "no_data"}
        
        if ind.get("error"):
            ind = {
                "ema_fast": 0, "ema_slow": 0, "rsi": 50, "atr": 0, "atr_pct": 0,
                "adx": 0, "vwap": 0, "close": df["close"].iloc[-1],
                "macro_trend_ok": False, "tick_size": 0.01
            }

        score = 1 
        if ind["ema_fast"] > ind["ema_slow"]: 
            score += 50
        if 30 <= ind["rsi"] <= 70: 
            score += 20
        
        ind["sector"] = config.SECTOR_MAP.get(sym, "Elite")
        scored.append((score, sym))
        indicators[sym] = ind
        
        # Mensagem de sucesso apenas na primeira carga
        with _first_build_lock:
            if not _first_build_done:
                logger.info(f"✅ Ativo Carregado: {sym} | Score: {score}")

    scored.sort(reverse=True, key=lambda x: x[0])
    selected_top = [s for _, s in scored[:15]]
    
    return indicators, selected_top
# =========================
# FILTROS COMUNS
# =========================
def additional_filters_ok(symbol: str) -> bool:
    df = utils.safe_copy_rates(symbol, TIMEFRAME_BASE, 100)
    if df is None or len(df) < 20:
        return False

    avg_vol = utils.get_avg_volume(df, window=20)

    min_vol = config.MIN_AVG_VOLUME_20
    if utils.is_power_hour():
        min_vol = int(min_vol * 0.6)  # flexibiliza liquidez na power-hour

    if avg_vol < min_vol:
        push_panel_alert(
        f"⚠️ {symbol} rejeitado: Volume médio baixo "
        f"({avg_vol:,.0f} < {min_vol:,.0f})",
        "INFO"
    )
    return False

    gap = utils.get_open_gap(symbol, TIMEFRAME_BASE)
    if gap is not None and gap > config.MAX_GAP_OPEN_PCT:
        push_panel_alert(f"⚠️ {symbol} rejeitado: Gap de abertura alto ({gap:.2f}% > {config.MAX_GAP_OPEN_PCT*100:.0f}%)", "INFO")
        return False

    # NOVO: Filtro de spread
    with mt5_lock:
        tick = mt5.symbol_info_tick(symbol)
    
    if not tick or tick.ask <= 0 or tick.bid <= 0:
        return False
    
    spread_pct = (tick.ask - tick.bid) / tick.bid * 100
    
    # Limites por horário
    if utils.is_power_hour():
        max_spread = 0.5  # 0.5% no fechamento (maior tolerância)
    else:
        max_spread = 0.3  # 0.3% no intraday normal
    
    if spread_pct > max_spread:
        push_panel_alert(
            f"⚠️ {symbol} rejeitado: Spread alto ({spread_pct:.2f}% > {max_spread}%)",
            "INFO"
        )
        return False
    
    return True

def get_sector_counts() -> defaultdict:
    with utils.mt5_lock:
        positions = mt5.positions_get() or []
    counts = defaultdict(int)
    for p in positions:
        sector = config.SECTOR_MAP.get(p.symbol, "UNKNOWN")
        counts[sector] += 1
    return counts

# =========================
# GESTÃO DE POSIÇÕES
# =========================
def modify_sl(symbol: str, ticket: int, new_sl: float):
    with utils.mt5_lock:
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
    with utils.mt5_lock:
        pos = mt5.positions_get(ticket=ticket)
        if not pos:
            return
        pos = pos[0]
    
    side = "BUY" if pos.type == mt5.POSITION_TYPE_BUY else "SELL"
    entry = pos.price_open

    profit_loss_points = (price - entry) if side == "BUY" else (entry - price)
    profit_loss_money = profit_loss_points * volume
    pl_pct = (profit_loss_money / (entry * volume)) * 100 if volume > 0 else 0

    # === REGISTRO DO RESULTADO PARA LOSS STREAK ===
    utils.record_trade_outcome(symbol, profit_loss_money)

    # === APRENDIZADO ML: Usa indicadores da ENTRADA (não da saída) ===
    with entry_indicators_lock:
        ind_at_entry = entry_indicators.get(symbol)

    if ind_at_entry:
        # Dados históricos para features avançadas (se o ML usar df)
        df = utils.safe_copy_rates(symbol, utils.TIMEFRAME_BASE, 100)

        # Registra o trade no ensemble com indicadores reais da entrada
        ml_optimizer.record_trade(
            symbol=symbol,
            ind=ind_at_entry,
            profit_pct=pl_pct / 100,
            df=df if df is not None else pd.DataFrame()
        )

        # Atualiza Q-Learning com reward baseado no resultado
        reward = pl_pct / 10  # Amplificação leve para aprendizado mais rápido
        current_ind = current_indicators.get(symbol, {})
        ml_optimizer.update_qlearning(reward, current_ind)

        # Limpa memória após registrar
        with entry_indicators_lock:
            entry_indicators.pop(symbol, None)

        push_alert(f"🧠 Trade registrado no ML para {symbol} | P&L: {pl_pct:+.2f}% | Usando indicadores de entrada", "INFO")
    else:
        # Caso raro: bot reiniciado ou entrada antiga sem registro
        logger.warning(f"⚠️ Indicadores de entrada não encontrados para {symbol}. Registrando com indicadores atuais (menos preciso).")
        df = utils.safe_copy_rates(symbol, utils.TIMEFRAME_BASE, 100)
        current_ind = current_indicators.get(symbol, {})
        ml_optimizer.record_trade(
            symbol=symbol,
            ind=current_ind,
            profit_pct=pl_pct / 100,
            df=df if df is not None else pd.DataFrame()
        )

    # === ENVIO DA ORDEM DE FECHAMENTO ===
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
            send_telegram_exit(
                symbol=symbol,
                side=side,
                volume=volume,
                entry_price=entry,
                exit_price=price,
                profit_loss=profit_loss_money,
                reason=reason
            )
        except Exception as e:
            logger.warning(f"Erro Telegram saída: {e}")
    else:
        push_alert(f"❌ Falha ao fechar {symbol}: {result.comment if result else 'Desconhecido'}", "WARNING")

def close_all_positions(reason: str = "Fechamento diário"):
    with utils.mt5_lock:
        positions = mt5.positions_get() or []
    
    if not positions:
        return
    
    for pos in positions:
        with utils.mt5_lock:
            tick = mt5.symbol_info_tick(pos.symbol)
        if not tick:
            continue
        price = tick.bid if pos.type == mt5.POSITION_TYPE_BUY else tick.ask
        close_position(pos.symbol, pos.ticket, pos.volume, price, reason=reason)
    
    push_alert(f"🔒 Todas as posições fechadas: {reason}", "INFO")

def manage_positions_advanced():
    with utils.mt5_lock:
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
        with current_indicators_lock:
            ind = current_indicators.get(symbol, {})
        
        close_by = datetime.strptime(config.CLOSE_ALL_BY, "%H:%M").time()
        if current_time.time() >= close_by:
            close_position(symbol, ticket, volume_total, current_price, reason="Day Close Forced")
            continue

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
    # Exemplo de uso ao fechar trade:
def on_trade_closed(symbol, profit_pct, score_log_at_entry):
    optimizer = AdaptiveWeightOptimizer()
    optimizer.record_trade(score_log_at_entry, profit_pct)

# =========================
# ENTRADA COM PYRAMIDING
# =========================
def try_enter_position(symbol: str, intended_side: str):
    """Lógica de filtragem final, cálculo de lote B3 e envio de ordem."""
    
    # 1. Validação de Horário e Conta
    if not is_trading_time_allowed(new_entry=True):
        return

    with utils.mt5_lock:
        acc = mt5.account_info()
    if not acc: return

    # 2. Recupera indicadores com Lock
    with current_indicators_lock:
        ind = current_indicators.get(symbol)
    
    if not ind or ind.get("atr", 0) <= 0:
        return

    # 3. Filtros Técnicos Adicionais (RSI / Filtros customizados)
    params = optimized_params.get(symbol, {})
    rsi_low = params.get("rsi_low", 35)
    rsi_high = params.get("rsi_high", 70)
    
    if not (rsi_low <= ind["rsi"] <= rsi_high):
        return

    if not additional_filters_ok(symbol):
        return
    # === BLOQUEIO POR DESEMPENHO (LOSS STREAK) ===
    blocked, reason = utils.is_symbol_blocked(symbol)
    if blocked:
        push_alert(f"⛔ {symbol} bloqueado: {reason}", "WARNING")
        return
    # 4. Gestão de Risco e Cálculo de Volume (Lote B3)
        # Risco base
    risk_pct = config.RISK_PER_TRADE_PCT

    # Ajuste por Portfolio Heat
    heat = get_portfolio_heat()
    if heat > 0.7:
        risk_pct *= 0.5
        push_alert(f"🔥 Portfolio Heat alto ({heat:.2f}) → Risco reduzido 50%", "WARNING")
    elif heat > 0.5:
        risk_pct *= 0.75
    
    # Se for pirâmide, ajusta o risco
    with utils.mt5_lock:
        existing = mt5.positions_get(symbol=symbol)
    
    leg = 0
    if not existing:
        risk_pct *= config.PYRAMID_RISK_SPLIT[0]
        leg = 1
    else:
        # Lógica de validação de pirâmide (distância ATR, etc)
        # Se não passar nos requisitos de pirâmide, retorna
        risk_pct *= config.PYRAMID_RISK_SPLIT[1]
        leg = 2
        # (Aqui entrariam os requisitos de ADX/Lucro que você já tem no código)

    volume_raw = utils.calculate_position_size_atr(acc.equity, risk_pct, ind["atr"])
    
    # ARREDONDAMENTO B3: Garante lotes de 100 (ou 1 se for fracionário)
    # Para mercado padrão (PETR4, VALE3), usamos múltiplos de 100.
    volume = round(volume_raw / 100.0) * 100.0
    
    if volume < 100: # Se o risco for tão pequeno que não compra 1 lote, aborta
        return

    # 5. Definição de Stop Loss e Take Profit
    df_history = utils.safe_copy_rates(symbol, TIMEFRAME_BASE, 100)
    sl = utils.calculate_smart_sl(symbol, ind["close"], intended_side, ind["atr"], df_history)
    
    if not sl: return
    
    distancia = abs(ind["close"] - sl)
    tp = ind["close"] + (distancia * 2) if intended_side == "BUY" else ind["close"] - (distancia * 2)

        # === SINAL ML (Q-Learning) ===
    ml_signal = ml_optimizer.get_ml_signal(ind)
    if ml_signal == "HOLD":
        return
    if ml_signal == "SELL" and intended_side == "BUY":  # Evita conflito
        return
    if ml_signal == "BUY" and intended_side == "SELL":
        return

    # 6. Envio da Ordem
    result = utils.send_order_with_sl_tp(symbol, intended_side, volume, sl, tp)
    
    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        _symbol_pyramid_leg[symbol] = {"leg": leg, "entry_time": datetime.now()}
        last_entry_attempt[symbol] = datetime.now()
        push_alert(f"✅ ORDEM EXECUTADA: {intended_side} {symbol} Vol:{volume}", "INFO")
        from telegram_async import send_async
        send_async(msg)
        # Salva os indicadores usados na entrada para aprendizado futuro
        with entry_indicators_lock:
            entry_indicators[symbol] = ind.copy()  # cópia profunda dos indicadores
        push_alert(f"📊 Indicadores de entrada salvos para {symbol} (ML learning)", "INFO")
    else:
        reason = result.comment if result else "Erro MT5"
        last_failure_reason[symbol] = reason

# =========================
# CIRCUIT BREAKER
# =========================
def check_for_circuit_breaker():
    global trading_paused, daily_max_equity, equity_inicio_dia, last_reset_day

    with utils.mt5_lock:
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

def get_portfolio_heat() -> float:
    """Calcula quão 'quente' está a carteira (0.0 = fria, 1.0 = superaquecida)"""
    with utils.mt5_lock:
        positions = mt5.positions_get() or []
    if len(positions) < 2:
        return 0.0

    symbols = [p.symbol for p in positions]

    # 1. Correlação média
    total_corr = 0.0
    count = 0
    for sym1 in symbols:
        for sym2 in symbols:
            if sym1 >= sym2:
                continue
            corr = utils.calculate_correlation_matrix([sym1, sym2]).get(sym1, {}).get(sym2, 0)
            total_corr += abs(corr)
            count += 1
    avg_corr = total_corr / count if count > 0 else 0.0

    # 2. Concentração setorial (HHI)
    equity = mt5.account_info().equity
    sector_exp = utils.calculate_sector_exposure_pct(equity)
    hhi = sum(exp ** 2 for exp in sector_exp.values())

    # 3. Volatilidade agregada
    total_atr_pct = sum(
        current_indicators.get(p.symbol, {}).get("atr_real", 0)
        for p in positions
    ) / len(positions) if positions else 0

    heat = (avg_corr * 0.4) + (hhi * 0.3) + (min(total_atr_pct / 10.0, 1.0) * 0.3)
    return round(min(heat, 1.0), 3)

# =========================
# HORÁRIO DE TRADING
# =========================
def is_trading_time_allowed(new_entry: bool = True) -> bool:
    now = datetime.now().time()
    start = datetime.strptime(config.TRADING_START, "%H:%M").time()
    no_entry = datetime.strptime(config.NO_ENTRY_AFTER, "%H:%M").time()
    force_close = datetime.strptime(config.CLOSE_ALL_BY, "%H:%M").time()
    
    if now < start:
        return False
    
    if new_entry and now > no_entry:
        push_alert("⚠️ Fora do horário para NOVAS entradas", "WARNING")
        return False
    
    if now >= force_close:
        close_all_positions(reason="Fechamento forçado diário (evitar overnight)")
        return False
    
    return True

# =========================
# RELATÓRIO DIÁRIO
# =========================
def daily_report():
    while True:
        now = datetime.now()
        if now.hour == 18 and now.minute < 5:
            with utils.mt5_lock:
                acc = mt5.account_info()
                positions_count = mt5.positions_total()
            if acc and equity_inicio_dia > 0:
                pnl_day = acc.equity - equity_inicio_dia
                msg = f"📊 <b>RELATÓRIO DIÁRIO XP3 - {now.strftime('%d/%m/%Y')}</b>\n\n"
                msg += f"Equity: R${acc.equity:,.2f}\n"
                msg += f"PnL do dia: <b>{pnl_day:+.2f}</b>\n"
                msg += f"Posições abertas: {positions_count}\n"
                msg += f"Status: {'PAUSADO' if trading_paused else 'ATIVO'}"

                try:
                    bot = get_telegram_bot()
                    bot.send_message(config.TELEGRAM_CHAT_ID, msg, parse_mode="HTML")
                    print("✅ Relatório diário enviado!")
                except Exception as e:
                    logger.warning(f"Erro ao enviar relatório: {e}")
            time.sleep(300)
        time.sleep(60)

# =========================
# DASHBOARD
# =========================
def render_panel_enhanced():
    clear_screen()
    
    with utils.mt5_lock:
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
    print(f"║ Posições: {mt5.positions_total()}/{config.MAX_SYMBOLS}  |  Risco/trade: {utils.get_current_risk_pct()*100:.2f}% {' '*35}║")
    print(f"╠{'═' * 96}╣")
    print("╠══════════ 🔧 STATUS DO SISTEMA ══════════╣")
    print(f"║ Correlação: {'✅ OK' if last_correlation_update else '⚠️ Nunca atualizada'} {' '*60}║")
    print(f"║ Pesos adaptativos: {'✅ Carregados' if utils.symbol_weights else '⚠️ Vazios'} {' '*56}║")
    print(f"║ Regime mercado: {detect_market_regime()} {' '*70}║")
    print(f"║ Power Hour: {'🔥 ATIVA' if utils.is_power_hour() else '—'} {' '*76}║")
    heat = get_portfolio_heat()
    heat_color = C_RED if heat > 0.7 else C_YELLOW if heat > 0.5 else C_GREEN
    print(f"║ Portfolio Heat: {heat_color}{heat:.3f}{C_RESET} {' '*70}║")
    
    # ML Status
    ml_trades = len(ml_optimizer.history) if hasattr(ml_optimizer, 'history') else 0
    epsilon = getattr(ml_optimizer, 'epsilon', 0.0)
    print(f"║ ML Trades: {ml_trades} | Epsilon: {epsilon:.3f} | Q-Table: {'Carregada' if os.path.exists('qtable.npy') else 'Nova'} {' '*40}║")
    
    # Últimos alertas
    print(f"╠{'═' * 96}╣")
    print(f"║ {C_YELLOW}🚨 ÚLTIMOS ALERTAS{C_RESET}{' '*76}║")
    with alerts_lock:
        recent = list(alerts)[:5]
    if not recent:
        print(f"║   {'(nenhum)':^92} ║")
    else:
        for _, msg in recent:
            msg_trunc = (msg[:90] + '...') if len(msg) > 90 else msg
            print(f"║   {msg_trunc:<92} ║")
    print(f"╠{'═' * 96}╣")

    # POSIÇÕES ABERTAS
    print(f"║ {C_GREEN}💼 POSIÇÕES ABERTAS ({mt5.positions_total()}){C_RESET}{' '*68}║")
    positions = mt5.positions_get() or []
    if not positions:
        print(f"║   {'(nenhuma posição aberta)':^92} ║")
    else:
        header = f"{'SYM':<6} {'DIR':<4} {'VOL':<8} {'ENTRY':<10} {'ATUAL':<10} {'P&L R$':<12} {'%':<7} {'STATUS':<15}"
        print(f"║ {header} ║")
        print(f"║ {'─'*6} {'─'*4} {'─'*8} {'─'*10} {'─'*10} {'─'*12} {'─'*7} {'─'*15} ║")
        for p in positions:
            tick = mt5.symbol_info_tick(p.symbol)
            if not tick:
                continue
            side = "BUY" if p.type == mt5.POSITION_TYPE_BUY else "SELL"
            current_price = tick.bid if p.type == mt5.POSITION_TYPE_BUY else tick.ask
            pct = (current_price - p.price_open)/p.price_open*100 if side=="BUY" else (p.price_open - current_price)/p.price_open*100
            ind = current_indicators.get(p.symbol, {})
            profit_atr = abs(current_price - p.price_open) / ind.get("atr", 0.01) if ind.get("atr", 0.01) > 0 else 0
            status = "Trailing" if profit_atr >= 1.5 else "Breakeven" if profit_atr >= 1.0 else "Aguardando"
            line = f"{p.symbol:<6} {side:<4} {p.volume:>7.0f} {p.price_open:>9.2f} {current_price:>9.2f} {p.profit:>+11.2f} {pct:>+6.1f}% {status:<15}"
            print(f"║ {line} ║")
    print(f"╠{'═' * 96}╣")

    # TOP 15
    print(f"║ {C_YELLOW}📊 TOP 15 ELITE{C_RESET}{' '*78}║")
    print(f"║ {'RK':<3} {'SYM':<6} {'SCORE':<6} {'DIR':<4} {'RSI':<5} {'ATR%':<6} {'HORA':<6} {'PREÇO':<8} {'CORR':<6} {'SETOR':<15} {'STATUS':<12} {'MOTIVO':<20} ║")
    print(f"║ {'─'*3} {'─'*6} {'─'*6} {'─'*4} {'─'*5} {'─'*6} {'─'*6} {'─'*8} {'─'*6} {'─'*15} {'─'*12} {'─'*20} ║")

    with top15_lock:
        top15_symbols = list(current_top15)

    if not top15_symbols:
        print(f"║   {'TOP15 vazio ou não carregado':^92} ║")
    else:
        positions_symbols = [p.symbol for p in mt5.positions_get() or []]
        current_sectors = get_sector_counts()

        for rank, sym in enumerate(top15_symbols, 1):
            with current_indicators_lock:
                ind = current_indicators.get(sym) or {}
            
            score = utils.calculate_signal_score(ind)
            long_signal = score >= config.MIN_SIGNAL_SCORE and ind.get("ema_fast", 0) > ind.get("ema_slow", 0)
            short_signal = score >= config.MIN_SIGNAL_SCORE and ind.get("ema_fast", 0) < ind.get("ema_slow", 0)
            dir_arrow = "↑" if long_signal else "↓" if short_signal else "-"
            side = "BUY" if long_signal else "SELL" if short_signal else None

            # Último candle
            df_last = utils.safe_copy_rates(sym, TIMEFRAME_BASE, 1)
            last_time = df_last.index[-1].strftime("%H:%M") if df_last is not None and not df_last.empty else "N/A"
            price_str = f"{ind.get('close', 0):.2f}" if ind.get('close') else "-"

            # Correlação
            avg_corr = get_average_correlation_with_portfolio(sym, positions_symbols) if config.ENABLE_CORRELATION_FILTER else 0.0
            corr_str = f"{avg_corr:.2f}" if avg_corr > 0 else "-"
            corr_color = C_RED if avg_corr > 0.7 else C_YELLOW if avg_corr > 0.5 else C_GREEN

            # Setor (truncado se necessário)
            sector_raw = config.SECTOR_MAP.get(sym, "UNKNOWN")
            sector = sector_raw[:14] + "…" if len(sector_raw) > 14 else sector_raw

            # Status e motivo
            if sym in positions_symbols:
                status = "✔️ ABERTO"
                status_color = C_GREEN
                motive = "Posição ativa"
            elif not additional_filters_ok(sym):
                status = "⏸️ FILTRO"
                status_color = C_RED
                motive = "Vol/Gap/Spread"
            elif not macro_trend_ok(sym, side):
                status = "⏸️ MACRO"
                status_color = C_RED
                motive = "Contra tendência"
            elif current_sectors.get(config.SECTOR_MAP.get(sym, "UNKNOWN"), 0) >= config.MAX_PER_SECTOR:
                status = "⏸️ SETOR"
                status_color = C_RED
                motive = "Limite setor"
            elif len(positions_symbols) >= config.MAX_SYMBOLS:
                status = "⏸️ CHEIO"
                status_color = C_RED
                motive = "Carteira cheia"
            elif score < config.MIN_SIGNAL_SCORE:
                status = "⏸️ SCORE"
                status_color = C_RED
                motive = f"Score {score:.1f}"
            else:
                status = "🟢 PRONTO"
                status_color = C_GREEN
                motive = "Sinal forte"

            # Motivo resumido do score_log
            log = ind.get("score_log", {})
            if log and "PRONTO" in status:
                parts = [f"{k}:{v}" for k, v in log.items() if abs(v) >= 10]
                motive = " | ".join(parts[:3])
                if len(parts) > 3:
                    motive += " | ..."
            
            motive_display = (motive[:19] + "…") if len(motive) > 19 else motive

            line = f"{rank:<3} {sym:<6} {score:>5.1f} {dir_arrow:<4} {ind.get('rsi',0):>4.1f} " \
                   f"{min(round(ind.get('atr_real',0)*5.3,2),9.99):>5.2f} {last_time:<6} {price_str:<8} " \
                   f"{corr_color}{corr_str:<6}{C_RESET} {sector:<15} {status_color}{status:<12}{C_RESET} {motive_display:<20}"
            print(f"║ {line} ║")

    print(f"╚{'═' * 96}╝")
# =========================
# FAST LOOP
# =========================
def fast_loop():
    """Loop principal: Processa sinais, faz a gestão e dispara ordens."""
    global current_indicators, current_top15, trading_paused
    
    logger.info("⚙️ Fast Loop iniciado.")
    
    while True:
        try:
            # 1. Verifica se o trading está pausado pelo Circuit Breaker
            if trading_paused:
                time.sleep(5)
                continue

            # 2. Verifica horário de negociação
            if not is_trading_time_allowed(new_entry=False):
                time.sleep(30)
                continue

            # 3. Reconstrói o Top 15 e calcula indicadores
            # build_portfolio_and_top15 já deve retornar os dados processados
            new_indicators, new_top15 = build_portfolio_and_top15()
            
            if new_indicators:
                with current_indicators_lock:
                    current_indicators = new_indicators
            
            if new_top15:
                with top15_lock:
                    current_top15 = new_top15

            # 4. Gestão de Posições (Trailing Stop, Break-even, etc.)
            manage_positions_advanced()

            # 5. Processamento de Sinais para Entrada
            with top15_lock:
                symbols_to_scan = list(current_top15)

            for sym in symbols_to_scan:
                with current_indicators_lock:
                    ind_data = current_indicators.get(sym)
                
                if not ind_data or ind_data.get("error"):
                    continue

                # Cálculo do Score e Direção
                score = utils.calculate_signal_score(ind_data)
                
                if score >= config.MIN_SIGNAL_SCORE:
                    side = "BUY" if ind_data["ema_fast"] > ind_data["ema_slow"] else "SELL"
                    
                    # Verifica se já está posicionado antes de tentar entrar
                    with utils.mt5_lock:
                        existing_pos = mt5.positions_get(symbol=sym)
                    
                    if not existing_pos:
                        try_enter_position(sym, side)
                    else:
                        # Se já estiver posicionado, a lógica de Pirâmide é tratada 
                        # dentro da try_enter_position ou manage_positions
                        try_enter_position(sym, side) 

            # 6. Verifica Circuit Breaker
            check_for_circuit_breaker()

            time.sleep(1) # Pulsação do bot

        except Exception as e:
            logger.error(f"Erro crítico no fast_loop: {e}", exc_info=True)
            time.sleep(10)

def maybe_send_eod_report():
    global _last_eod_report_date

    if not config.EOD_REPORT_ENABLED:
        return

    now = datetime.now()
    today = now.date()

    eod_time = datetime.strptime(config.EOD_REPORT_TIME, "%H:%M").time()

    # ainda não chegou no horário
    if now.time() < eod_time:
        return

    # já enviou hoje
    if _last_eod_report_date == today:
        return

    utils.send_telegram_eod_report()
    _last_eod_report_date = today


# =========================
# MAIN
# =========================
def main():
    """Ponto de entrada: Inicializa MT5, carrega dados e dispara as threads."""
    clear_screen()
    print(f"{C_CYAN}Iniciando XP3 PRO BOT B3...{C_RESET}")
    
    if not mt5.initialize():
        logger.critical("❌ Falha ao inicializar MetaTrader 5.")
        return
    
    utils.load_loss_streak_data()
    # 1. Preparação Inicial
    load_optimized_params()
    utils.load_adaptive_weights()
    
    # 2. Carga Inicial de Dados (Síncrona para evitar que o bot comece vazio)
    logger.info("Carregando Top 15 inicial...")
    try:
        ind, top = build_portfolio_and_top15()
        with current_indicators_lock:
            global current_indicators
            current_indicators = ind
        with top15_lock:
            global current_top15
            current_top15 = top
    except Exception as e:
        logger.error(f"Erro na carga inicial: {e}")

    # 3. Disparo das Threads de Trabalho
    # Loop de negociação
    t_fast = threading.Thread(target=fast_loop, daemon=True, name="FastLoop")
    t_fast.start()
    
    # Relatório de fim de dia
    t_daily = threading.Thread(target=daily_report, daemon=True, name="DailyReport")
    t_daily.start()
    
    # Atualizador de correlação (cada 30 min)
    t_corr = threading.Thread(target=correlation_updater_thread, daemon=True, name="CorrUpdater")
    t_corr.start()

    logger.info("🚀 Threads iniciadas com sucesso.")
    utils.save_loss_streak_data()
    # 4. Loop de Interface (Roda na Main Thread)
    try:
        while True:
            render_panel_enhanced()
            time.sleep(2) # Atualiza a tela a cada 2 segundos
    except KeyboardInterrupt:
        logger.info("Bot encerrado pelo usuário.")
    finally:
        mt5.shutdown()

if __name__ == "__main__":
    main()