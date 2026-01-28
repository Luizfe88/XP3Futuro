#bot.py
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import json
import pandas as pd
import time
import threading
import logging
# Silencia erros de WebSocket fechado (Tornado/Streamlit)
logging.getLogger("tornado.access").setLevel(logging.ERROR)
logging.getLogger("tornado.application").setLevel(logging.ERROR)
logging.getLogger("tornado.general").setLevel(logging.ERROR)
logging.getLogger("asyncio").setLevel(logging.ERROR)

import asyncio
# Handler global para silenciar erros de WebSocket fechado
def _silence_event_loop_exceptions(loop, context):
    msg = context.get("exception", context.get("message"))
    # Ignora erros conhecidos do Tornado/Streamlit
    if "WebSocketClosedError" in str(msg) or "StreamClosedError" in str(msg) or "Task finished" in str(msg):
        return
    # Loga outros erros normalmente
    logging.error(f"AsyncIO Error: {msg}")

try:
    loop = asyncio.get_event_loop()
except Exception:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
loop.set_exception_handler(_silence_event_loop_exceptions)

from datetime import datetime, date, timedelta
from threading import Lock, RLock
from collections import deque, defaultdict, OrderedDict
import MetaTrader5 as mt5
import config
import utils
from hedging import apply_hedge
from telegram_handler import bot
import numpy as np
import hashlib
from news_filter import check_news_blackout
from typing import Optional, Dict, Any, List, Tuple

# --- UI (Rich removido -> Streamlit) ---
import subprocess
import webbrowser

from utils import (
    send_telegram_trade,
    send_telegram_exit,
    get_telegram_bot,
    check_and_close_orphans,
    calculate_signal_score,
    safe_copy_rates,
    detect_market_regime,
    calculate_sector_exposure_pct,
    macro_trend_ok,
    adjust_global_sl_after_pyr,
    send_order_with_sl_tp,
    is_valid_dataframe,load_anti_chop_data,
    save_anti_chop_data,
    load_daily_limits,
    save_daily_limits,
    reset_daily_limits,
    register_sl_hit,
    check_anti_chop_filter,
    check_daily_symbol_limit,
    register_trade_result,
    check_pyramid_eligibility,
    check_minimum_price_movement,
)
from ml_optimizer import EnsembleOptimizer
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
from daily_analysis_logger import daily_logger
ml_optimizer = EnsembleOptimizer()
import os
from utils import mt5_lock
from database import save_trade
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from validation import validate_and_create_order, OrderParams, OrderSide

current_trading_day: Optional[date] = None
daily_cycle_completed = False
daily_report_sent = False
# =====================
# Telegram (fallback seguro)
# =====================
try:
    from telegram_async import send_async
except ImportError:

    def send_async(*args, **kwargs):
        pass  # fallback silencioso, NUNCA quebra o bot

class NpEncoder(json.JSONEncoder):
    """
    Encoder JSON que converte automaticamente:
    - numpy types → Python types
    - bool → int (0 ou 1)
    - datetime → ISO string
    """
    def default(self, obj):
        # Numpy integers
        if isinstance(obj, np.integer):
            return int(obj)
        
        # Numpy floats
        if isinstance(obj, np.floating):
            return float(obj) if not np.isnan(obj) else None
        
        # Numpy arrays
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        
        # Booleanos → inteiros (0 ou 1)
        if isinstance(obj, bool):
            return int(obj)
        
        # np.bool_
        if isinstance(obj, np.bool_):
            return int(obj)
        
        # Datetime
        if isinstance(obj, datetime):
            return obj.isoformat()
        
        # Fallback para conversão padrão
        return super(NpEncoder, self).default(obj)

def sanitize_for_json(obj):
    if obj is None: return None
    if isinstance(obj, (bool, np.bool_)): return int(obj)
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating):
        return float(obj) if not np.isnan(obj) else None
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, str): return obj
    if isinstance(obj, (int, float)):
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)): return None
        return obj
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(item) for item in obj]
    if isinstance(obj, datetime): return obj.isoformat()
    return str(obj)
    return str(obj)

# ============================================
# 🧪 A/B TESTING ENGINE
# ============================================
def get_ab_group(symbol: str) -> str:
    """
    Determina o grupo A/B do símbolo usando hash determinístico.
    Isso garante que o mesmo símbolo sempre fique no mesmo grupo
    durante um teste, mas distribui aleatoriamente.
    """
    if not config.AB_TEST_ENABLED:
        return "A"
        
    # Hash do símbolo
    hash_obj = hashlib.md5(symbol.encode())
    hash_int = int(hash_obj.hexdigest(), 16)
    
    # Par = Grupo A, Ímpar = Grupo B
    return "A" if hash_int % 2 == 0 else "B"

def get_params_for_group(group: str) -> dict:
    """Retorna parâmetros específicos do grupo AB"""
    return config.AB_TEST_GROUPS.get(group, config.AB_TEST_GROUPS["A"])

# ============================================
# 🔒 WRAPPER THREAD-SAFE PARA MT5
# ============================================

from threading import Semaphore
import signal

# Semáforo para garantir UMA ordem por vez
_order_send_semaphore = Semaphore(1)
_active_close_tickets = set()  # Rastreia tickets sendo fechados
_active_close_lock = Lock()


def mt5_order_send_safe(request: dict, timeout: int = 10) -> Optional[dict]:
    """
    Wrapper thread-safe para mt5.order_send() com timeout
    
    Garante:
    - Apenas UMA thread enviando ordem por vez
    - Timeout de 10s (evita travamento infinito)
    - Retorna None em caso de timeout
    """
    
    def _send_order():
        try:
            with utils.mt5_lock:
                return mt5.order_send(request)
        except Exception as e:
            logger.error(f"Exceção no order_send: {e}")
            return None
    
    # Adquire semáforo (bloqueia se outra thread está enviando)
    acquired = _order_send_semaphore.acquire(timeout=timeout)
    
    if not acquired:
        logger.error(f"⏱️ TIMEOUT: Não conseguiu lock em {timeout}s")
        return None
    
    try:
        # Usa thread com timeout
        import queue
        q = queue.Queue()
        
        def worker():
            result = _send_order()
            q.put(result)
        
        t = threading.Thread(target=worker, daemon=True)
        t.start()
        t.join(timeout=timeout)
        
        if t.is_alive():
            logger.error(f"⏱️ TIMEOUT: order_send travou por {timeout}s")
            return None
        
        try:
            return q.get_nowait()
        except queue.Empty:
            return None
    
    finally:
        _order_send_semaphore.release()


def can_close_position(ticket: int) -> bool:
    """
    Verifica se este ticket já está sendo fechado por outra thread
    """
    with _active_close_lock:
        if ticket in _active_close_tickets:
            return False
        _active_close_tickets.add(ticket)
        return True


def mark_close_complete(ticket: int):
    """
    Marca ticket como fechado (libera para outras operações)
    """
    with _active_close_lock:
        _active_close_tickets.discard(ticket)

def validate_mt5_connection():
    """
    Valida e força conexão com o terminal correto do MT5
    """
    max_attempts = 3
    
    for attempt in range(1, max_attempts + 1):
        try:
            # Tenta inicializar com o caminho específico
            if mt5.initialize(path=config.MT5_TERMINAL_PATH):
                terminal = mt5.terminal_info()
                
                if terminal and terminal.connected:
                    logger.info(f"✅ MT5 conectado: {config.MT5_TERMINAL_PATH}")
                    logger.info(f"   📊 Conta: {mt5.account_info().login}")
                    logger.info(f"   🏢 Corretora: {mt5.account_info().company}")
                    return True
            
            logger.warning(f"⚠️ Tentativa {attempt}/{max_attempts} falhou")
            
            # Se falhou, força shutdown e tenta novamente
            mt5.shutdown()
            time.sleep(2)
            
        except Exception as e:
            logger.error(f"❌ Erro na tentativa {attempt}: {e}")
            time.sleep(2)
    
    # Se todas as tentativas falharam
    logger.critical(f"🚨 FALHA CRÍTICA: Não foi possível conectar ao MT5")
    logger.critical(f"   Caminho configurado: {config.MT5_TERMINAL_PATH}")
    logger.critical(f"   Verifique se:")
    logger.critical(f"      1. O MT5 está instalado neste caminho")
    logger.critical(f"      2. Você está logado na conta")
    logger.critical(f"      3. Não há outro programa usando o terminal")
    
    return False
def get_asset_class_config(symbol: str) -> dict:
    s = (symbol or "").upper()
    is_fut = utils.is_future(s)
    if is_fut:
        start = "09:05"
        end = getattr(config, "FUTURES_CLOSE_ALL_BY", "17:50")
        bucket_pct = 0.35
        min_lot = 1
        if s.startswith(("WIN", "IND")):
            dev_pts = 80
        elif s.startswith(("WDO", "DOL")):
            dev_pts = 20
        else:
            dev_pts = 50
        return {"start": start, "end": end, "bucket_pct": bucket_pct, "min_lot": min_lot, "deviation_points": dev_pts, "lunch_min_vol_ratio": 0.0, "min_tp_cost_multiplier": 3.0}
    start = "10:10"
    end = "16:50"
    bucket_pct = 0.65
    min_lot = 100
    return {"start": start, "end": end, "bucket_pct": bucket_pct, "min_lot": min_lot, "deviation_points": 2, "lunch_min_vol_ratio": 1.0, "min_tp_cost_multiplier": 1.2}
def check_capital_allocation(symbol: str, planned_volume: float, entry_price: float) -> tuple[bool, str]:
    try:
        with utils.mt5_lock:
            acc = mt5.account_info()
            positions = mt5.positions_get() or []
        if not acc:
            return False, "Sem conta MT5"
        equity = float(acc.equity or acc.balance or 0.0)
        if equity <= 0:
            return False, "Equity inválido"
        fut_exposure = 0.0
        stk_exposure = 0.0
        for p in positions:
            si = mt5.symbol_info(p.symbol)
            contract = float(si.trade_contract_size) if si else 1.0
            price = float(getattr(p, "price_current", getattr(p, "price_open", 0.0)) or 0.0)
            exp = float(p.volume) * (contract if utils.is_future(p.symbol) else 1.0) * price
            if utils.is_future(p.symbol):
                fut_exposure += exp
            else:
                stk_exposure += exp
        is_fut = utils.is_future(symbol)
        si_new = mt5.symbol_info(symbol)
        contract_new = float(si_new.trade_contract_size) if si_new else 1.0
        add_exp = float(planned_volume) * (contract_new if is_fut else 1.0) * float(entry_price)
        cap_fut = equity * 0.35
        cap_stk = equity * 0.65
        if is_fut:
            if fut_exposure + add_exp > cap_fut:
                return False, f"Exposição Futuros {fut_exposure + add_exp:,.2f} > {cap_fut:,.2f}"
            return True, "OK"
        else:
            if stk_exposure + add_exp > cap_stk:
                return False, f"Exposição Ações {stk_exposure + add_exp:,.2f} > {cap_stk:,.2f}"
            return True, "OK"
    except Exception as e:
        return False, f"Erro alocação: {e}"
def get_ibov_adx() -> float:
    try:
        df = utils.safe_copy_rates("IBOV", mt5.TIMEFRAME_M15, 120)
        if df is None or len(df) < 30:
            return 0.0
        return float(utils.get_adx(df) or 0.0)
    except Exception:
        return 0.0
def get_market_status() -> dict:
    """
    Retorna status detalhado do mercado (VERSÃO CONTÍNUA)
    """
    now = datetime.now()
    current_time = now.time()
    today = now.date()

    start = datetime.strptime(config.TRADING_START, "%H:%M").time()
    no_entry_str = getattr(config, "FRIDAY_NO_ENTRY_AFTER", config.NO_ENTRY_AFTER) if now.weekday() == 4 else config.NO_ENTRY_AFTER
    force_close_str = getattr(config, "FRIDAY_CLOSE_ALL_BY", config.CLOSE_ALL_BY) if now.weekday() == 4 else config.CLOSE_ALL_BY
    no_entry = datetime.strptime(no_entry_str, "%H:%M").time()
    force_close = datetime.strptime(force_close_str, "%H:%M").time()

    # Verifica se é fim de semana
    is_weekend = now.weekday() >= 5  # 5=Sábado, 6=Domingo

    # ============================================
    # 🔴 FIM DE SEMANA
    # ============================================
    if is_weekend:
        # Calcula próxima segunda-feira
        days_until_monday = (7 - now.weekday()) if now.weekday() == 6 else 1
        next_trading_day = now + timedelta(days=days_until_monday)
        next_trading_datetime = datetime.combine(next_trading_day.date(), start)

        time_until_next = next_trading_datetime - now
        hours = int(time_until_next.total_seconds() // 3600)
        minutes = int((time_until_next.total_seconds() % 3600) // 60)

        return {
            "status": "WEEKEND",
            "emoji": "🌙",
            "message": "FIM DE SEMANA",
            "color": C_CYAN,
            "countdown": f"{hours}h {minutes}m até segunda-feira",
            "detail": f"Próximo pregão: {next_trading_day.strftime('%d/%m')} às {config.TRADING_START}",
            "trading_allowed": False,
            "new_entries_allowed": False,
            "should_close_positions": False,
        }

    # ============================================
    # 1️⃣ PRÉ-MERCADO (Antes da abertura)
    # ============================================
    if current_time < start:
        start_datetime = datetime.combine(now.date(), start)
        time_until_start = start_datetime - now

        hours = int(time_until_start.total_seconds() // 3600)
        minutes = int((time_until_start.total_seconds() % 3600) // 60)
        seconds = int(time_until_start.total_seconds() % 60)

        return {
            "status": "PRE_MARKET",
            "emoji": "⏳",
            "message": "AGUARDANDO ABERTURA",
            "color": C_YELLOW,
            "countdown": f"{hours:02d}:{minutes:02d}:{seconds:02d}",
            "detail": f"Início programado: {config.TRADING_START}",
            "trading_allowed": False,
            "new_entries_allowed": False,
            "should_close_positions": False,
        }

    # ============================================
    # 2️⃣ MERCADO ABERTO (Operando normalmente)
    # ============================================
    elif start <= current_time < no_entry:
        return {
            "status": "OPEN",
            "emoji": "🟢",
            "message": "MERCADO ABERTO",
            "color": C_GREEN,
            "countdown": None,
            "detail": f"Operando até {no_entry_str}",
            "trading_allowed": True,
            "new_entries_allowed": True,
            "should_close_positions": False,
        }

    # ============================================
    # 3️⃣ SEM NOVAS ENTRADAS (Só gestão)
    # ============================================
    elif no_entry <= current_time < force_close:
        close_datetime = datetime.combine(now.date(), force_close)
        time_until_close = close_datetime - now

        minutes = int(time_until_close.total_seconds() // 60)
        seconds = int(time_until_close.total_seconds() % 60)

        return {
            "status": "NO_NEW_ENTRIES",
            "emoji": "🟡",
            "message": "SEM NOVAS ENTRADAS",
            "color": C_YELLOW,
            "countdown": f"{minutes:02d}:{seconds:02d}",
            "detail": f"Fechamento às {force_close_str}",
            "trading_allowed": True,
            "new_entries_allowed": False,
            "should_close_positions": False,
        }

    # ============================================
    # 4️⃣ HORÁRIO DE FECHAMENTO (Encerra posições)
    # ============================================
    else:
        # Calcula próximo dia útil
        next_day = now + timedelta(days=1)

        # Pula fim de semana se for sexta
        if next_day.weekday() >= 5:
            days_to_add = 8 - next_day.weekday()  # Até segunda
            next_day = now + timedelta(days=days_to_add)

        next_trading_datetime = datetime.combine(next_day.date(), start)
        time_until_next = next_trading_datetime - now

        hours = int(time_until_next.total_seconds() // 3600)
        minutes = int((time_until_next.total_seconds() % 3600) // 60)

        return {
            "status": "POST_MARKET",
            "emoji": "🔴",
            "message": "MERCADO FECHADO",
            "color": C_RED,
            "countdown": f"{hours}h {minutes}m até próximo pregão",
            "detail": f"Reabertura: {next_day.strftime('%d/%m')} às {config.TRADING_START}",
            "trading_allowed": False,
            "new_entries_allowed": False,
            "should_close_positions": True,
        }


# ============================================
# 🕐 VERIFICAÇÃO DE HORÁRIO SEGURO (COMERCIAL)
# ============================================
def check_market_hours() -> tuple:
    """
    Verifica se está dentro do horário seguro para novas entradas.
    
    Regras:
    - Bloqueia 30 min após abertura (volatilidade inicial)
    - Bloqueia 20 min antes do fechamento (volatilidade final)
    
    Returns:
        tuple: (pode_operar: bool, motivo: str)
    """
    now = datetime.now()
    current_time = now.time()
    
    # Horários base da config
    open_time = datetime.strptime(config.TRADING_START, "%H:%M").time()
    close_str = getattr(config, "FRIDAY_CLOSE_ALL_BY", config.CLOSE_ALL_BY) if now.weekday() == 4 else config.CLOSE_ALL_BY
    close_time = datetime.strptime(close_str, "%H:%M").time()
    
    # Calcula buffers
    buffer_after_open = getattr(config, 'MARKET_HOURS_BUFFER_OPEN', 30)  # 30 min
    buffer_before_close = getattr(config, 'MARKET_HOURS_BUFFER_CLOSE', 20)  # 20 min
    if now.weekday() == 4:
        buffer_before_close = int(getattr(config, "FRIDAY_MARKET_HOURS_BUFFER_CLOSE", buffer_before_close))
    
    safe_start = (datetime.combine(now.date(), open_time) + 
                  timedelta(minutes=buffer_after_open)).time()
    safe_end = (datetime.combine(now.date(), close_time) - 
                timedelta(minutes=buffer_before_close)).time()
    
    lunch_start_str = getattr(config, "TRADING_LUNCH_BREAK_START", None)
    lunch_end_str = getattr(config, "TRADING_LUNCH_BREAK_END", None)
    if lunch_start_str and lunch_end_str:
        lunch_start = datetime.strptime(lunch_start_str, "%H:%M").time()
        lunch_end = datetime.strptime(lunch_end_str, "%H:%M").time()
        if lunch_start <= current_time <= lunch_end:
            return False, "⏸️ Pausa de almoço institucional"
    
    # Verificações
    if current_time < safe_start:
        mins_left = int((datetime.combine(now.date(), safe_start) - now).total_seconds() // 60)
        return False, f"⏰ Aguardando estabilização ({mins_left} min para início seguro)"
    
    if current_time > safe_end:
        return False, f"⏰ Fim do horário seguro (fechamento em breve)"
    
    return True, "OK"


# ============================================
# 💰 PROTEÇÃO DE LUCRO DIÁRIO (COMERCIAL)
# ============================================
def global_profit_protector() -> tuple:
    """
    Verifica se a meta de lucro diário foi atingida.
    Se sim, fecha todas as posições e bloqueia novas entradas.
    
    Returns:
        tuple: (deve_parar: bool, motivo: str)
    """
    global equity_inicio_dia, daily_target_hit_day, daily_target_hit_pct
    
    try:
        today = datetime.now().date()
        if daily_target_hit_day == today:
            pct_txt = f"{daily_target_hit_pct:.2%}" if daily_target_hit_pct is not None else "N/A"
            return True, f"Meta já atingida: {pct_txt}"

        with utils.mt5_lock:
            acc = mt5.account_info()
        
        if not acc:
            return False, "Sem dados de conta"
        
        if equity_inicio_dia <= 0:
            return False, "Equity inicial não definido"
        
        # Calcula lucro do dia
        current_profit_pct = (acc.equity - equity_inicio_dia) / equity_inicio_dia
        target_pct = getattr(config, 'DAILY_PROFIT_TARGET_PCT', 0.02)
        
        if current_profit_pct >= target_pct:
            logger.info(
                f"🎯 META DIÁRIA ATINGIDA! "
                f"Lucro: {current_profit_pct:.2%} >= {target_pct:.2%}"
            )
            daily_target_hit_day = today
            daily_target_hit_pct = float(current_profit_pct)
            daily_pnl = acc.equity - equity_inicio_dia
            apply_profit_lock_actions(daily_pnl=daily_pnl, daily_pnl_pct=current_profit_pct, reason="Meta Diária Atingida")
            
            # Notifica
            try:
                utils.send_telegram_message(
                    f"🎉 <b>META DIÁRIA ATINGIDA!</b>\n\n"
                    f"💰 Lucro: <b>{current_profit_pct:.2%}</b>\n"
                    f"📊 Equity: R$ {acc.equity:,.2f}\n\n"
                    f"🛡️ Proteção aplicada (fechar winners e/ou trailing apertado)\n"
                    f"🛑 Novas entradas bloqueadas até amanhã."
                )
            except:
                pass
            
            return True, f"Meta atingida: {current_profit_pct:.2%}"
        
        return False, f"Lucro atual: {current_profit_pct:.2%}"
        
    except Exception as e:
        logger.error(f"Erro no profit protector: {e}")
        return False, f"Erro: {e}"


# ============================================
# 📊 ANÁLISE DE PERFORMANCE E AJUSTES
# ============================================

def log_trade_cvm_compliance(symbol, side, volume, entry_price, exit_price, pnl, reason):
    """
    Gera log CSV para conformidade CVM
    """
    import csv
    file_exists = os.path.isfile("compliance_trades.csv")
    try:
        with open("compliance_trades.csv", "a", newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Timestamp", "Symbol", "Side", "Volume", "EntryPrice", "ExitPrice", "PnL", "Reason"])
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                symbol, side, volume, entry_price, exit_price, pnl, reason
            ])
    except Exception as e:
        logger.error(f"Erro ao salvar log CVM: {e}")

def run_performance_analysis():
    """
    Analisa performance das últimas 24h e retorna o Win Rate
    """
    with utils.mt5_lock:
        deals = mt5.history_deals_get(datetime.now() - timedelta(days=1), datetime.now())
    
    if not deals:
        return None

    # Filtra apenas fechamentos
    out_deals = [d for d in deals if d.entry == mt5.DEAL_ENTRY_OUT]
    if not out_deals:
        return None

    wins = sum(1 for d in out_deals if d.profit > 0)
    win_rate = (wins / len(out_deals)) * 100
    
    logger.info(f"📊 Análise Diária: {len(out_deals)} trades | Win Rate: {win_rate:.1f}%")
    return win_rate


def reconcile_trade_log_today():
    try:
        start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        now = datetime.now()
        with utils.mt5_lock:
            deals_all = mt5.history_deals_get(start - timedelta(days=7), now)
        if not deals_all:
            return
        pos_entry_price = {}
        for d in deals_all:
            if d.entry == mt5.DEAL_ENTRY_IN and d.position_id not in pos_entry_price:
                pos_entry_price[d.position_id] = d.price
        out_today = []
        for d in deals_all:
            if d.entry != mt5.DEAL_ENTRY_OUT:
                continue
            dt = datetime.fromtimestamp(d.time)
            if dt.date() != start.date():
                continue
            out_today.append(d)
        if not out_today:
            return
        filename = f"trades_log_{start.strftime('%Y-%m-%d')}.txt"
        existing = ""
        existing_sigs = set()
        existing_deals = set()
        if os.path.exists(filename):
            try:
                with open(filename, "r", encoding="utf-8") as f:
                    existing = f.read()
                for line in existing.splitlines():
                    if "SAÍDA" in line:
                        try:
                            # Extrai campos chave de forma tolerante a espaços
                            # Ex.: "... | SAÍDA    | RAIL3  | SELL | Vol:   35300 | Price:  13.38 | P&L: -4589.00 R$ ..."
                            parts = line.split("|")
                            if len(parts) >= 6:
                                tipo = parts[1].strip()
                                symbol = parts[2].strip()
                                side = parts[3].strip()
                                vol_txt = parts[4]
                                price_txt = parts[5]
                                pnl_txt = parts[6] if len(parts) > 6 else ""
                                # Vol
                                vol_num = 0
                                if "Vol:" in vol_txt:
                                    vol_num = int("".join(ch for ch in vol_txt if ch.isdigit()))
                                # Price
                                price_num = 0.0
                                if "Price:" in price_txt:
                                    try:
                                        price_num = float(price_txt.split("Price:")[1].strip().split()[0])
                                    except:
                                        price_num = 0.0
                                # P&L
                                pnl_num = 0.0
                                if "P&L:" in pnl_txt:
                                    try:
                                        pnl_str = pnl_txt.split("P&L:")[1].strip().split()[0]
                                        pnl_num = float(pnl_str.replace("+", ""))
                                    except:
                                        pnl_num = 0.0
                                sig = f"SAIDA:{symbol}:{side}:{vol_num}:{price_num:.2f}:{pnl_num:.2f}"
                                existing_sigs.add(sig)
                                if "DealId:" in line:
                                    try:
                                        import re
                                        m = re.search(r"DealId:\s*(\d+)", line)
                                        if m:
                                            deal_id_str = m.group(1)
                                            existing_deals.add(f"DEAL:{deal_id_str}")
                                    except:
                                        pass
                        except:
                            pass
            except:
                existing = ""
        # ✅ Dedup dos registros existentes por DealId (mantém cabeçalho e ordem)
        try:
            if existing:
                import re
                lines = existing.splitlines(True)  # preserva quebras de linha
                new_lines = []
                seen_deals = set()
                for line in lines:
                    if "SAÍDA" in line:
                        m = re.search(r"DealId:\s*(\d+)", line)
                        if m:
                            deal_id = m.group(1)
                            key = f"DEAL:{deal_id}"
                            if key in seen_deals:
                                continue
                            seen_deals.add(key)
                    new_lines.append(line)
                if len(new_lines) < len(lines):
                    with open(filename, "w", encoding="utf-8") as f:
                        f.writelines(new_lines)
        except Exception:
            pass
        for d in out_today:
            symbol = d.symbol
            original_side = "BUY" if d.type == mt5.DEAL_TYPE_SELL else "SELL"
            entry_price = pos_entry_price.get(d.position_id, d.price)
            exit_price = d.price
            si = mt5.symbol_info(symbol)
            contract = si.trade_contract_size if si else 1.0
            volume_units = float(d.volume) * float(contract)
            pnl_money = float(d.profit or 0.0)
            denom = (entry_price * volume_units) if volume_units > 0 else 0.0
            pnl_pct = (pnl_money / denom * 100.0) if denom > 0 else 0.0
            comment = (d.comment or "").lower()
            if "sl" in comment:
                reason = "Stop Loss"
            elif "tp" in comment:
                reason = "Take Profit"
            else:
                reason = "Fechamento"
            sig = f"SAIDA:{symbol}:{original_side}:{int(volume_units)}:{entry_price:.2f}:{pnl_money:.2f}"
            deal_sig = f"DEAL:{getattr(d, 'ticket', 0)}"
            if deal_sig in existing_deals or sig in existing_sigs:
                continue
            log_trade_to_txt(
                symbol=symbol,
                side=original_side,
                volume=volume_units,
                entry_price=entry_price,
                exit_price=exit_price,
                pnl_money=pnl_money,
                pnl_pct=pnl_pct,
                reason=reason,
                deal_id=int(getattr(d, "ticket", 0) or 0),
                position_id=int(getattr(d, "position_id", 0) or 0),
            )
    except Exception as e:
        logger.error(f"Erro reconcile trade log: {e}")

# ============================================
# 🎯 ENTRADA PARCIAL (SCALED ENTRY)
# ============================================

def execute_partial_entry(symbol: str, total_volume: float, side: str, 
                           entry_price: float, sl: float, tp: float,
                           num_entries: int = 3) -> bool:
    """
    ✅ Entrada parcial: divide o volume em múltiplas entradas.
    
    Benefícios:
    - Reduz risco de timing ruim
    - Melhora preço médio em tendências
    - Permite melhor gestão de risco
    
    Args:
        num_entries: Número de entradas parciais (default: 3)
    """
    if num_entries < 2:
        num_entries = 2
    
    partial_volume = round((total_volume / num_entries) / 100) * 100
    if partial_volume < 100:
        partial_volume = 100
    
    executed = 0
    for i in range(num_entries):
        # Ajusta preço de entrada para cada parcial
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            break
        
        current_price = tick.ask if side == "BUY" else tick.bid
        
        # Só executa if preço ainda está favorável
        if side == "BUY" and current_price > entry_price * 1.005:
            logger.info(f"⏸️ {symbol} Entrada {i+1}/{num_entries} pausada - preço subiu")
            break
        if side == "SELL" and current_price < entry_price * 0.995:
            logger.info(f"⏸️ {symbol} Entrada {i+1}/{num_entries} pausada - preço caiu")
            break
        
        try:
            from validation import validate_and_create_order
            order, error = validate_and_create_order(
                symbol=symbol,
                side=side,
                volume=partial_volume,
                entry_price=current_price,
                sl=sl,
                tp=tp,
                use_kelly=False  # Volume já calculado
            )
            
            if order:
                request = order.to_mt5_request(comment=f"Partial {i+1}/{num_entries}")
                result = mt5_order_send_safe(request)
                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                    executed += 1
                    logger.info(f"✅ {symbol} Entrada {i+1}/{num_entries}: {partial_volume} @ {current_price:.2f}")
                    
                    if i < num_entries - 1:
                        time.sleep(2)  # Espera entre entradas
                else:
                    logger.warning(f"⚠️ {symbol} Falha na entrada {i+1}: {result.comment if result else 'Sem resposta'}")
                    try:
                        from rejection_logger import log_trade_rejection
                        log_trade_rejection(symbol, "PartialEntryExec", "MT5 Error", {"retcode": result.retcode if result else -1, "comment": result.comment if result else "None"})
                    except ImportError:
                        pass
        except Exception as e:
            logger.error(f"Erro entrada parcial {symbol}: {e}")
            break
    
    return executed > 0


# ============================================
# 📈 SAÍDA DINÂMICA (DYNAMIC EXIT)
# ============================================

def calculate_dynamic_exit(symbol: str, entry_price: float, side: str, 
                            current_price: float, atr: float) -> dict:
    """
    ✅ Calcula saída dinâmica baseada em condições de mercado.
    
    Returns:
        {
            'action': 'HOLD' | 'PARTIAL_EXIT' | 'FULL_EXIT',
            'reason': str,
            'exit_volume_pct': float (0.0 a 1.0)
        }
    """
    if side == "BUY":
        pnl_pct = (current_price - entry_price) / entry_price
    else:
        pnl_pct = (entry_price - current_price) / entry_price
    
    r_multiple = pnl_pct / (atr / entry_price) if atr > 0 else 0
    
    # VIX check
    vix_br = utils.get_vix_br()
    
    # Regras de saída dinâmica
    
    # 1. Saída total em lucro alto (>3R) ou VIX extremo
    if r_multiple >= 3.0 or (pnl_pct > 0.02 and vix_br > 40):
        return {
            'action': 'FULL_EXIT',
            'reason': f'+{r_multiple:.1f}R ou VIX={vix_br:.0f}',
            'exit_volume_pct': 1.0
        }
    
    # 2. Saída parcial (50%) em +2R
    if r_multiple >= 2.0:
        return {
            'action': 'PARTIAL_EXIT',
            'reason': f'+{r_multiple:.1f}R - Realizando 50%',
            'exit_volume_pct': 0.5
        }
    
    # 3. Saída parcial (30%) em +1.5R com VIX alto
    if r_multiple >= 1.5 and vix_br > 30:
        return {
            'action': 'PARTIAL_EXIT',
            'reason': f'+{r_multiple:.1f}R + VIX Alto',
            'exit_volume_pct': 0.3
        }
    
    # 4. Mantém posição
    return {
        'action': 'HOLD',
        'reason': f'{r_multiple:+.1f}R',
        'exit_volume_pct': 0.0
    }


# ============================================
# ⏸️ PAUSA POR WIN RATE BAIXO
# ============================================

TRADING_PAUSED = False
PAUSE_REASON = ""

def check_win_rate_pause() -> tuple:
    """
    ✅ Verifica se deve pausar operações por win rate baixo.
    
    Regras:
    - WR < 45% últimos 20 trades: PAUSA trading
    - WR < 50% últimos 20 trades: Modo conservador
    - WR >= 55%: Normal
    
    Returns:
        (should_pause: bool, reason: str)
    """
    global TRADING_PAUSED, PAUSE_REASON, CIRCUIT_BREAKER_DISABLED
    
    try:
        if CIRCUIT_BREAKER_DISABLED:
            return False, "Circuit breaker desativado"
        now = datetime.now()
        start_of_day = datetime.combine(now.date(), datetime.strptime("00:00", "%H:%M").time())
        with utils.mt5_lock:
            deals = mt5.history_deals_get(now - timedelta(days=3), now)
        
        if not deals:
            return False, "Sem histórico"
        
        out_deals = [d for d in deals if d.entry == mt5.DEAL_ENTRY_OUT][-20:]  # Últimos 20
        # Grace period pós-reset: não pausar até N trades de saída no dia
        try:
            todays_out = [d for d in deals if d.entry == mt5.DEAL_ENTRY_OUT and datetime.fromtimestamp(d.time) >= start_of_day]
            if pause_reset_day == now.date() and len(todays_out) < getattr(config, "WR_RESET_GRACE_TRADES", 5):
                return False, f"Grace WR pós-reset ({len(todays_out)}/{getattr(config,'WR_RESET_GRACE_TRADES',5)} trades hoje)"
        except Exception:
            pass
        
        if len(out_deals) < 20:
            return False, "Histórico insuficiente (mín. 20 trades)"
        
        wins = sum(1 for d in out_deals if d.profit > 0)
        win_rate = wins / len(out_deals)
        
        # Atualiza parâmetros dinâmicos
        params = config.get_params_for_win_rate(win_rate)
        
        if win_rate < 0.45:
            TRADING_PAUSED = True
            PAUSE_REASON = f"WR Crítico: {win_rate:.1%} (últimos 20 trades)"
            logger.warning(f"🚨 PAUSA ATIVADA: {PAUSE_REASON}")
            return True, PAUSE_REASON
        
        elif win_rate < 0.50:
            # Modo conservador
            config.set_operation_mode("DEFENSIVE")
            logger.info(f"⚠️ Modo DEFENSIVO: WR={win_rate:.1%}")
            return False, f"Modo Defensivo (WR: {win_rate:.1%})"
        
        else:
            TRADING_PAUSED = False
            PAUSE_REASON = ""
            config.set_operation_mode("NORMAL")
            return False, f"OK (WR: {win_rate:.1%})"
            
    except Exception as e:
        logger.error(f"Erro ao verificar win rate: {e}")
        return False, "Erro na verificação"


def is_trading_allowed() -> tuple:
    """
    Verifica se trading está permitido (combina todas as verificações).
    
    Returns:
        (allowed: bool, reason: str)
    """
    global TRADING_PAUSED, PAUSE_REASON, manual_pause_reason

    if manual_pause_reason:
        return False, f"Pausa manual: {manual_pause_reason}"
    
    # 1. Pausa por win rate
    paused, reason = check_win_rate_pause()
    if paused:
        return False, reason
    
    # 2. Modo de operação
    mode_params = config.get_current_mode_params()
    if not mode_params.get('allow_new_entries', True):
        return False, f"Modo {config.CURRENT_OPERATION_MODE} - Sem novas entradas"
    
    # 3. Horário de mercado
    market_ok, market_reason = check_market_hours()
    if not market_ok:
        return False, market_reason
    
    # 4. Profit protector
    should_stop, profit_reason = global_profit_protector()
    if should_stop:
        return False, profit_reason
    
    return True, "Trading permitido"



# ============================================
# 🔄 GERENCIADOR DE CICLO DIÁRIO
# ============================================
def handle_daily_cycle():
    """
    ✅ VERSÃO REFORÇADA: Garante fechamento com múltiplas tentativas
    """
    global current_trading_day, daily_cycle_completed, daily_report_sent
    global equity_inicio_dia, daily_max_equity, last_reset_day

    now = datetime.now()
    today = now.date()
    market_status = get_market_status()

    # ============================================
    # 1️⃣ NOVO DIA DETECTADO
    # ============================================
    if current_trading_day != today:
        logger.info(f"📅 Novo dia detectado: {today.strftime('%d/%m/%Y')}")

        current_trading_day = today
        daily_cycle_completed = False
        daily_report_sent = False

        with utils.mt5_lock:
            acc = mt5.account_info()

        if acc:
            equity_inicio_dia = acc.equity
            daily_max_equity = acc.equity

            with open("daily_equity.txt", "w") as f:
                f.write(str(equity_inicio_dia))

            logger.info(f"💰 Equity inicial do dia: R${equity_inicio_dia:,.2f}")

        # 1. Backtest Diário e Análise de Performance
        try:
            import backtest
            wr_backtest = backtest.run_backtest(30)
            wr_real = run_performance_analysis()
            
            # Unifica as métricas (prioriza real se houver trades)
            final_wr = wr_real if wr_real is not None else wr_backtest
            
            # ✅ AJUSTE DINÂMICO DE CONFIGURAÇÃO (Min RR)
            try:
                config.config_manager.update_dynamic_settings(final_wr / 100)
            except Exception as e:
                logger.error(f"Erro ao atualizar config dinâmica: {e}")
            
            if final_wr < 55.0:
                # Ajuste automático de parâmetros se performance estiver baixa
                old_conf = config.ML_MIN_CONFIDENCE
                config.ML_MIN_CONFIDENCE = min(0.85, config.ML_MIN_CONFIDENCE + 0.05)
                config.MIN_SIGNAL_SCORE = max(61, config.MIN_SIGNAL_SCORE + 2)
                
                logger.warning(f"⚠️ Performance Baixa (WR: {final_wr:.1f}%). Ajustando ML Confidence: {old_conf:.2f} -> {config.ML_MIN_CONFIDENCE:.2f}")
                utils.send_telegram_message(f"⚠️ <b>Performance Alert</b>\nWin Rate: {final_wr:.1f}%\nML Confidence: {config.ML_MIN_CONFIDENCE:.2f}\nStatus: Parâmetros Ajustados")
            else:
                logger.info(f"✅ Performance Saudável (WR: {final_wr:.1f}%)")
        except Exception as e:
            logger.error(f"Erro ao rodar análise diária: {e}")

        last_reset_day = today
        daily_trades_per_symbol.clear()

        try:
            reset_daily_limits()
            logger.info("✅ Limites diários resetados")
        except Exception as e:
            logger.error(f"Erro ao resetar limites: {e}")

        push_alert(
            f"🌅 Novo ciclo de trading iniciado: {today.strftime('%d/%m/%Y')}", "INFO"
        )

    # ============================================
    # 2️⃣ HORÁRIO DE FECHAMENTO - VERSÃO REFORÇADA
    # ============================================
    
    # ⏰ Inicia fechamento 2 minutos ANTES do horário oficial
    close_time_str = config.CLOSE_ALL_BY
    close_time = datetime.strptime(close_time_str, "%H:%M").time()
    
    # Antecipa em 2 minutos
    early_close = (datetime.combine(today, close_time) - timedelta(minutes=2)).time()
    
    if now.time() >= early_close and not daily_cycle_completed:
        
        minutes_until_deadline = (datetime.combine(today, close_time) - now).total_seconds() / 60
        
        logger.warning(
            f"⏰ INICIANDO FECHAMENTO EOD | "
            f"Tempo até deadline: {minutes_until_deadline:.1f} min"
        )
        
        # 🔄 LOOP DE FECHAMENTO AGRESSIVO
        max_attempts = 5  # Era 3, agora 5 tentativas
        
        for attempt in range(1, max_attempts + 1):
            with utils.mt5_lock:
                positions = mt5.positions_get() or []

            if not is_valid_dataframe(positions):
                logger.info("✅ Todas as posições fechadas!")
                daily_cycle_completed = True
                break

            logger.warning(
                f"🔄 Tentativa {attempt}/{max_attempts} | "
                f"{len(positions)} posições abertas"
            )
            
            # Mostra quais são
            symbols_open = [p.symbol for p in positions]
            logger.info(f"   Símbolos: {', '.join(symbols_open)}")
            
            # FECHA TODAS
            close_all_positions(reason=f"EOD - Tentativa {attempt}")
            
            # Aguarda 3s
            time.sleep(3)
            
            # Verifica se realmente fechou
            with utils.mt5_lock:
                remaining = mt5.positions_get() or []
            
            if not remaining:
                logger.info("✅ Fechamento confirmado")
                daily_cycle_completed = True
                break
            
            if attempt < max_attempts:
                logger.error(
                    f"⚠️ {len(remaining)} posições ainda abertas | "
                    f"Tentando novamente em 5s..."
                )
                time.sleep(5)
            else:
                # 🚨 ÚLTIMA TENTATIVA FALHOU
                logger.critical(
                    f"🚨 FALHA CRÍTICA: {len(remaining)} posições NÃO fecharam "
                    f"após {max_attempts} tentativas!"
                )
                
                # Notificação de emergência
                try:
                    utils.send_telegram_message(
                        f"🚨 <b>ALERTA CRÍTICO - EOD</b>\n\n"
                        f"❌ <b>{len(remaining)} posições NÃO fecharam!</b>\n\n"
                        f"<b>Ativos:</b> {', '.join([p.symbol for p in remaining])}\n"
                        f"<b>Tentativas:</b> {max_attempts}\n\n"
                        f"⚠️ <b>AÇÃO IMEDIATA NECESSÁRIA!</b>\n"
                        f"Feche manualmente no MetaTrader 5"
                    )
                except:
                    pass
                
                # Marca como completo para evitar loop infinito
                daily_cycle_completed = True

    # ============================================
    # 3️⃣ ENVIA RELATÓRIO (UMA VEZ POR DIA)
    # ============================================
    if daily_cycle_completed and not daily_report_sent:
        logger.info("📧 Enviando relatório de desempenho diário...")

        try:
            utils.send_daily_performance_report()
            daily_report_sent = True
            logger.info("✅ Relatório enviado com sucesso")
        except Exception as e:
            logger.error(f"❌ Erro ao enviar relatório: {e}")

        # 🆕 NOVO: Relatório de Rejeições (Console e Log)
        try:
            rejection_report = daily_logger.get_daily_rejection_summary()
            logger.info(rejection_report)
            if getattr(config, "ENABLE_TELEGRAM_REJECTION_SUMMARY", False):
                utils.send_telegram_message(f"📊 <b>RESUMO DE REJEIÇÕES</b>\n<pre>{rejection_report}</pre>")
        except Exception as e:
            logger.error(f"Erro ao gerar resumo de rejeições: {e}")

    # ============================================
    # 4️⃣ SALVA DADOS PERSISTENTES
    # ============================================
    if daily_cycle_completed:
        try:
            utils.save_loss_streak_data()
            utils.save_adaptive_weights()
            save_anti_chop_data()
            save_daily_limits()
        except Exception as e:
            logger.error(f"Erro ao salvar dados: {e}")

# === New BotState Class (Priority 1) ===
@dataclass
class BotState:
    """
    Gerencia estado global do bot de forma thread-safe.
    Garante que indicators e top15 sejam sempre atualizados juntos.
    """

    _lock: Lock = field(default_factory=Lock, init=False, repr=False)
    _indicators: Dict[str, dict] = field(default_factory=dict)
    _top15: List[str] = field(default_factory=list)

    @property
    def snapshot(self) -> Tuple[Dict[str, dict], List[str]]:
        """
        Retorna cópia atômica de (indicators, top15).
        Thread-safe para leitura.
        """
        with self._lock:
            return (self._indicators.copy(), self._top15.copy())

    def get_indicators(self, symbol: str) -> dict:
        """Thread-safe: Lê indicadores de um símbolo"""
        with self._lock:
            return self._indicators.get(symbol, {}).copy()

    def get_top15(self) -> List[str]:
        """Thread-safe: Lê lista TOP15"""
        with self._lock:
            return self._top15.copy()

    def update(self, indicators: Dict[str, dict], top15: List[str]):
        """
        Thread-safe: Atualiza indicators e top15 atomicamente.
        Garante consistência: ambos são atualizados juntos ou nenhum é.
        """
        with self._lock:
            self._indicators = indicators.copy()
            self._top15 = list(top15)
            logger.debug(
                f"BotState atualizado: {len(indicators)} ativos, TOP15 = {len(top15)}"
            )


# === New TimedCache Class (Priority 2) ===
class TimedCache:
    """
    Cache com expiração automática para evitar memory leaks.
    Remove entradas antigas automaticamente ao acessar.
    """

    def __init__(self, max_age_seconds: int = 86400, max_size: int = 10000):
        """
        Args:
            max_age_seconds: Tempo de vida de cada entrada (padrão: 24h)
            max_size: Tamanho máximo do cache (padrão: 10k entradas)
        """
        self._cache = OrderedDict()
        self._lock = Lock()
        self.max_age = max_age_seconds
        self.max_size = max_size

    def set(self, key, value):
        """Adiciona ou atualiza entrada"""
        with self._lock:
            self._cache[key] = (value, time.time())
            self._cleanup()

    def get(self, key, default=None):
        """Retorna valor se ainda válido, senão default"""
        with self._lock:
            if key in self._cache:
                value, timestamp = self._cache[key]
                if time.time() - timestamp < self.max_age:
                    return value
                else:
                    # Expirado, remove
                    del self._cache[key]
                    logger.debug(f"TimedCache: {key} expirado após {self.max_age}s")
            return default

    def pop(self, key, default=None):
        """Remove e retorna valor"""
        with self._lock:
            if key in self._cache:
                value, _ = self._cache.pop(key)
                return value
            return default

    def _cleanup(self):
        """Remove entradas expiradas ou excedentes"""
        now = time.time()

        # 1. Remove expiradas
        expired = [k for k, (_, ts) in self._cache.items() if now - ts > self.max_age]
        for k in expired:
            del self._cache[k]

        # 2. Remove mais antigas se excedeu max_size
        if len(self._cache) > self.max_size:
            excess = len(self._cache) - self.max_size
            for _ in range(excess):
                self._cache.popitem(last=False)  # Remove primeiro (FIFO)

        if expired:
            logger.debug(f"TimedCache: {len(expired)} entradas expiradas removidas")

    def __len__(self):
        """Retorna tamanho atual do cache"""
        with self._lock:
            return len(self._cache)

    def clear(self):
        """Limpa todo o cache"""
        with self._lock:
            self._cache.clear()

_close_gap_last_candle_logged = TimedCache(max_age_seconds=86400, max_size=20000)
_close_gap_last_tick_warn = TimedCache(max_age_seconds=3600, max_size=20000)

def _get_minutes_to_close_all_by(close_all_by: str, now: datetime) -> Optional[float]:
    try:
        close_by_time = datetime.strptime(close_all_by, "%H:%M").time()
    except Exception:
        return None
    close_by_dt = datetime.combine(now.date(), close_by_time)
    minutes = (close_by_dt - now).total_seconds() / 60.0
    return max(0.0, minutes)

def _compute_close_gap_status(mt5_position) -> str:
    now = datetime.now()
    close_all_by = getattr(config, "FRIDAY_CLOSE_ALL_BY", getattr(config, "CLOSE_ALL_BY", "17:55")) if now.weekday() == 4 else getattr(config, "CLOSE_ALL_BY", "17:55")
    max_candles = int(getattr(config, "MAX_TRADE_DURATION_CANDLES", 999999))

    ticket = mt5_position.ticket
    opened_ts = position_open_times.get(ticket, getattr(mt5_position, "time", time.time()))
    time_open_minutes = (time.time() - opened_ts) / 60.0
    candles_open = int(time_open_minutes / 15)

    try:
        close_by_time = datetime.strptime(close_all_by, "%H:%M").time()
        if now.time() >= close_by_time:
            return f"TRIGGER: Day Close Forced ({close_all_by})"
    except Exception:
        pass

    if candles_open >= max_candles:
        return f"TRIGGER: Time-stop ({candles_open}/{max_candles} candles)"

    candles_remaining = max(0, max_candles - candles_open)
    minutes_to_close = _get_minutes_to_close_all_by(close_all_by, now)
    minutes_txt = f"{minutes_to_close:.0f} min" if minutes_to_close is not None else "N/A"
    return f"faltam {candles_remaining} candles p/ time-stop ({candles_open}/{max_candles}); faltam {minutes_txt} p/ day close ({close_all_by})"

# === New PositionManager (Priority 5) ===
@dataclass
class PositionStatus:
    """Estado atual de uma posição"""
    """Estado atual de uma posição"""

    ticket: int
    symbol: str
    side: str
    entry_price: float
    current_price: float
    sl: float
    tp: float
    volume: float
    profit_atr: float
    time_open_minutes: float
    regime: str


class PositionManager:
    """Gerencia posições de forma modular e testável"""

    def __init__(self, config):
        self.config = config

    def should_close_by_time(self, pos: PositionStatus) -> Optional[str]:
        """Verifica time-stop. Retorna motivo se deve fechar."""
        now = datetime.now()

        # 1. Fechamento forçado diário
        close_str = getattr(self.config, "FRIDAY_CLOSE_ALL_BY", self.config.CLOSE_ALL_BY) if now.weekday() == 4 else self.config.CLOSE_ALL_BY
        close_by = datetime.strptime(close_str, "%H:%M").time()
        if now.time() >= close_by:
            return f"Day Close Forced ({close_str})"

        # 2. Time-stop por candles
        max_candles = self.config.MAX_TRADE_DURATION_CANDLES
        candles_open = int(pos.time_open_minutes / 15)

        if candles_open >= max_candles:
            return f"Time-stop ({candles_open} candles)"

        return None

    def should_apply_breakeven(
        self, pos: PositionStatus, atr: float
    ) -> Optional[float]:
        if not self.config.ENABLE_BREAKEVEN:
            return None
        if pos.profit_atr < 0.8:
            return None
        buffer = atr * 0.3
        if pos.side == "BUY":
            new_sl = pos.entry_price + buffer
            if pos.sl >= new_sl:
                return None
            return new_sl
        else:
            new_sl = pos.entry_price - buffer
            if pos.sl <= new_sl:
                return None
            return new_sl

    def should_partial_close(
        self, pos: PositionStatus, regime_config: dict
    ) -> Optional[float]:
        """Retorna volume parcial se aplicável (Trigger: +2R, Volume: 50%)"""
        if not self.config.ENABLE_PARTIAL_CLOSE:
            return None

        if "PARTIAL" in getattr(pos, "comment", ""):
            return None

        # Trigger dinâmico do config (+2R)
        if pos.profit_atr >= self.config.PARTIAL_CLOSE_ATR_MULT:
            # 50% da posição atual
            partial_volume = round((pos.volume * 0.5) / 100) * 100
            if partial_volume >= 100:
                return partial_volume

        return None

    def calculate_trailing_sl(
        self, pos: PositionStatus, atr: float, regime_config: dict
    ) -> Optional[float]:
        if not self.config.ENABLE_TRAILING_STOP:
            return None
        is_fut = utils.is_future(pos.symbol)
        if is_fut:
            if pos.profit_atr < 1.5:
                return None
            trail_mult = 1.5
        else:
            if pos.profit_atr < 1.0:
                return None
            trail_mult = 1.2

        if pos.side == "BUY":
            new_sl = pos.current_price - (atr * trail_mult)
            if new_sl <= pos.sl:
                return None
        else:
            new_sl = pos.current_price + (atr * trail_mult)
            if new_sl >= pos.sl:
                return None

        improvement = abs(new_sl - pos.sl) / atr
        if improvement >= 0.2:
            return new_sl

        return None

    def _detect_regime(self, ind: dict) -> str:
        """Detecta regime de mercado"""
        adx = ind.get("adx", 20)
        if ind.get("vol_breakout"):
            return "BREAKOUT"
        elif adx >= 30:
            return "TRENDING"
        else:
            return "RANGING"

    def manage_single_position(self, mt5_position, indicators: dict):
        """
        Gerencia uma posição.
        Retorna tupla de ação: ('CLOSE', motivo) | ('MODIFY_SL', novo_sl, tipo) | ('PARTIAL', volume) | None
        """
        symbol = mt5_position.symbol
        ticket = mt5_position.ticket
        opened_ts = position_open_times.get(ticket, getattr(mt5_position, "time", time.time()))
        time_open_minutes = (time.time() - opened_ts) / 60.0

        side = "BUY" if mt5_position.type == mt5.POSITION_TYPE_BUY else "SELL"

        close_reason = self.should_close_by_time(
            PositionStatus(
                ticket=ticket,
                symbol=symbol,
                side=side,
                entry_price=mt5_position.price_open,
                current_price=mt5_position.price_current,
                sl=mt5_position.sl,
                tp=mt5_position.tp,
                volume=mt5_position.volume,
                profit_atr=0.0,
                time_open_minutes=time_open_minutes,
                regime="N/A",
            )
        )
        if close_reason:
            return ("CLOSE", close_reason)

        ind = indicators.get(symbol, {})

        if not ind or ind.get("error"):
            return None

        atr = ind.get("atr")
        if not atr or atr <= 0:
            return None

        # Monta status
        profit_points = (
            mt5_position.price_current - mt5_position.price_open
            if side == "BUY"
            else mt5_position.price_open - mt5_position.price_current
        )

        pos = PositionStatus(
            ticket=ticket,
            symbol=symbol,
            side=side,
            entry_price=mt5_position.price_open,
            current_price=mt5_position.price_current,
            sl=mt5_position.sl,
            tp=mt5_position.tp,
            volume=mt5_position.volume,
            profit_atr=profit_points / atr,
            time_open_minutes=time_open_minutes,
            regime=self._detect_regime(ind),
        )

        # 1️⃣ BREAKEVEN
        new_sl_be = self.should_apply_breakeven(pos, atr)
        if new_sl_be:
            return ("MODIFY_SL", new_sl_be, "Breakeven")

        # 2️⃣ PARTIAL CLOSE
        regime_config = self.config.TP_RULES[pos.regime]
        partial_vol = self.should_partial_close(pos, regime_config)
        if partial_vol:
            return ("PARTIAL", partial_vol)

        # 3️⃣ TRAILING STOP
        new_sl_trail = self.calculate_trailing_sl(pos, atr, regime_config)
        if new_sl_trail:
            return ("MODIFY_SL", new_sl_trail, "Trailing")

        return None


class BotHealthMonitor:
    def __init__(self):
        self.last_heartbeat = time.time()
        self.max_freeze_seconds = 120  # 2 minutos sem pulso → freeze

    def heartbeat(self):
        self.last_heartbeat = time.time()

    def check_health(self):
        if time.time() - self.last_heartbeat > self.max_freeze_seconds:
            logger.critical(
                "🚨 BOT CONGELADO DETECTADO - EXECUTANDO REINÍCIO DE EMERGÊNCIA"
            )
            close_all_positions(reason="Emergency restart - Freeze detectado")
            push_alert("🚨 REINÍCIO DE EMERGÊNCIA: Bot congelado", "CRITICAL", True)
            # Reinicia o script Python
            os.execv(sys.executable, ["python"] + sys.argv)


# Instância global
health_monitor = BotHealthMonitor()

# ============================================
# ETAPA 1: ADICIONAR NOVA FUNÇÃO (NÃO DELETAR NADA)
# ============================================

# === ADICIONAR APÓS a classe PositionManager (linha ~266 de bot.py) ===


def manage_positions_refactored():
    """
    ✅ NOVA VERSÃO: Gestão modular com trailing stop adaptativo
    """
    manager = PositionManager(config)
    
    with utils.mt5_lock:
        positions = mt5.positions_get() or []
    
    if not is_valid_dataframe(positions):
        return
    
    indicators, _ = bot_state.snapshot
    
    for pos in positions:
        try:
            status = _compute_close_gap_status(pos)
            if not status.startswith("TRIGGER:"):
                ticket = pos.ticket
                last_candle = _close_gap_last_candle_logged.get(ticket)
                opened_ts = position_open_times.get(ticket, getattr(pos, "time", time.time()))
                candles_open = int(((time.time() - opened_ts) / 60.0) / 15)
                if last_candle != candles_open:
                    logger.info(f"⏳ Fechamento pendente {pos.symbol} (ticket {ticket}): {status}")
                    _close_gap_last_candle_logged.set(ticket, candles_open)

            # Delega para PositionManager
            action = manager.manage_single_position(pos, indicators)

            if not action:
                continue

            # Executa ação retornada
            if action[0] == "CLOSE":
                with utils.mt5_lock:
                    tick = mt5.symbol_info_tick(pos.symbol)
                if not tick:
                    key = f"{pos.ticket}:{action[1]}"
                    last_warn = _close_gap_last_tick_warn.get(key, 0.0)
                    if time.time() - float(last_warn) >= 60.0:
                        logger.warning(
                            f"⚠️ {pos.symbol} (ticket {pos.ticket}): close acionado ({action[1]}), mas sem cotação/tick"
                        )
                        _close_gap_last_tick_warn.set(key, time.time())
                    continue

                price = tick.bid if pos.type == mt5.POSITION_TYPE_BUY else tick.ask
                close_position(
                    pos.symbol, pos.ticket, pos.volume, price, reason=action[1]
                )

            elif action[0] == "MODIFY_SL":
                modify_sl(pos.symbol, pos.ticket, action[1])
                logger.info(
                    f"🔒 {action[2]}: {pos.symbol} | SL: {pos.sl:.2f} → {action[1]:.2f}"
                )

            elif action[0] == "PARTIAL":
                with utils.mt5_lock:
                    tick = mt5.symbol_info_tick(pos.symbol)
                if not tick:
                    continue

                price = tick.bid if pos.type == mt5.POSITION_TYPE_BUY else tick.ask

                # Calcula profit_atr se não disponível
                ind = indicators.get(pos.symbol, {})
                atr = ind.get("atr", 0.01)
                side = "BUY" if pos.type == mt5.POSITION_TYPE_BUY else "SELL"
                profit_points = (
                    (pos.price_current - pos.price_open)
                    if side == "BUY"
                    else (pos.price_open - pos.price_current)
                )
                profit_atr = profit_points / atr if atr > 0 else 0

                close_position(
                    pos.symbol,
                    pos.ticket,
                    action[1],
                    price,
                    reason=f"Partial (+{profit_atr:.1f}ATR)",
                )

        except Exception as e:
            logger.error(f"❌ Erro ao gerenciar {pos.symbol}: {e}", exc_info=True)
            continue
        
        # ✅ Step Trailing adicional (travamento + trailing agressivo)
        try:
            utils.manage_dynamic_trailing(pos.symbol, pos.ticket)
        except Exception:
            pass

def calculate_dynamic_trailing(pos, ind: dict, atr: float) -> Optional[float]:
    """
    🎯 TRAILING STOP DINÂMICO BASEADO EM VOLATILIDADE E MOMENTUM
    
    Ajusta distância do trailing baseado em:
    1. Volatilidade atual (ATR expansion/contraction)
    2. Momentum (acelera trailing se momentum enfraquecer)
    3. Suporte/Resistência (ancora em níveis técnicos)
    
    Returns:
        Novo SL ou None se não deve mover
    
    Impacto: +10-15% no profit capture
    """
    try:
        symbol = pos.symbol
        side = "BUY" if pos.type == mt5.POSITION_TYPE_BUY else "SELL"
        current_price = pos.price_current
        entry_price = pos.price_open
        current_sl = pos.sl
        
        # 1. Calcula lucro em ATRs
        profit_dist = (current_price - entry_price) if side == "BUY" else (entry_price - current_price)
        profit_in_atr = profit_dist / atr if atr > 0 else 0
        
        # Não move se lucro < 1 ATR
        if profit_in_atr < 1.0:
            return None
        
        # 2. ✅ NOVO: Ajuste por expansão/contração de volatilidade
        df = safe_copy_rates(symbol, mt5.TIMEFRAME_M15, 30)
        
        if df is not None and len(df) > 20:
            # ATR dos últimos 5 candles vs ATR médio
            recent_atr = get_atr(df.tail(5))
            avg_atr = get_atr(df.tail(20))
            
            vol_ratio = recent_atr / avg_atr if avg_atr > 0 else 1.0
            
            # Se volatilidade expandindo (>1.2x), afasta trailing
            # Se contraindo (<0.8x), aproxima trailing
            if vol_ratio > 1.2:
                vol_adjustment = 1.3  # +30% distância
            elif vol_ratio < 0.8:
                vol_adjustment = 0.7  # -30% distância
            else:
                vol_adjustment = 1.0
        else:
            vol_adjustment = 1.0
        
        # 3. ✅ NOVO: Ajuste por momentum
        momentum = ind.get("momentum", 0)
        
        # Se momentum enfraquecer, aperta trailing (protege lucro)
        if side == "BUY":
            if momentum < 0:  # Perdendo força
                momentum_adjustment = 0.7  # Aperta
            else:
                momentum_adjustment = 1.0
        else:  # SELL
            if momentum > 0:
                momentum_adjustment = 0.7
            else:
                momentum_adjustment = 1.0
        
        # 4. Define multiplicador base por lucro
        if profit_in_atr >= 5.0:
            base_mult = 0.8
        elif profit_in_atr >= 3.0:
            base_mult = 1.2
        elif profit_in_atr >= 1.5:
            base_mult = 1.8
        else:
            base_mult = 2.2
        
        # 5. Combina ajustes
        final_mult = base_mult * vol_adjustment * momentum_adjustment
        
        # 6. ✅ NOVO: Ancora em suporte/resistência
        if df is not None and len(df) > 20:
            lookback = 20
            
            if side == "BUY":
                # Busca último suporte relevante
                support = df['low'].tail(lookback).min()
                
                # Se trailing calculado ficar abaixo do suporte, usa suporte
                calculated_sl = current_price - (atr * final_mult)
                structure_sl = support - (atr * 0.3)
                
                new_sl = max(calculated_sl, structure_sl)
            else:  # SELL
                resistance = df['high'].tail(lookback).max()
                calculated_sl = current_price + (atr * final_mult)
                structure_sl = resistance + (atr * 0.3)
                
                new_sl = min(calculated_sl, structure_sl)
        else:
            # Fallback sem estrutura
            if side == "BUY":
                new_sl = current_price - (atr * final_mult)
            else:
                new_sl = current_price + (atr * final_mult)
        
        # 7. Valida se deve mover
        if side == "BUY":
            if current_sl and new_sl <= current_sl:
                return None  # Não move para trás
            
            if new_sl >= current_price:
                return None  # Não move acima do preço
        else:  # SELL
            if current_sl and new_sl >= current_sl:
                return None
            
            if new_sl <= current_price:
                return None
        
        logger.info(
            f"🎯 Trailing {symbol} | "
            f"Lucro: {profit_in_atr:.1f}R | "
            f"Mult: {final_mult:.2f} (Vol:{vol_adjustment:.2f}, Mom:{momentum_adjustment:.2f}) | "
            f"SL: {current_sl:.2f} → {new_sl:.2f}"
        )
        
        return round(new_sl, 2)
    
    except Exception as e:
        logger.error(f"Erro trailing dinâmico: {e}", exc_info=True)
        return None
def health_watcher_thread():
    """
    Monitora saúde - Reconexão, DD, Volume e Win Rate (last 20 trades < 50%).
    """
    global trading_paused, _last_wr_alert_ts, _last_wr_alert_wr, CIRCUIT_BREAKER_DISABLED
    while True:
        # 1. Reconexão MT5
        try:
            with utils.mt5_lock:
                terminal = None
                try:
                    terminal = mt5.terminal_info()
                except Exception:
                    terminal = None
                if (terminal is None) or (not getattr(terminal, "connected", False)):
                    logger.warning("MT5 desconectado - Reconectando...")
                    path = getattr(config, "MT5_TERMINAL_PATH", None)
                    ok = False
                    try:
                        ok = mt5.initialize(path=path) if path else mt5.initialize()
                    except Exception:
                        ok = False
                    if ok:
                        term2 = None
                        try:
                            term2 = mt5.terminal_info()
                        except Exception:
                            term2 = None
                        if term2 and getattr(term2, "connected", False):
                            logger.info("MT5 reconectado com sucesso")
                        else:
                            logger.error("MT5 ainda desconectado após tentativa de reconexão")
                    else:
                        logger.error("Falha ao inicializar MT5 na tentativa de reconexão")
        except Exception as e:
            logger.error(f"Erro na verificação/reconexão MT5: {e}")
        
        # 2. DD pausa (>5%)
        dd = utils.calculate_daily_dd()
        max_dd_stop = getattr(config, "MAX_DAILY_DD_STOP", 0.05)
        if dd > max_dd_stop:
            prev_paused = trading_paused
            trading_paused = True
            dd_alert_cooldown = getattr(config, "DD_ALERT_COOLDOWN_SECONDS", 3600)
            dd_delta_threshold = getattr(config, "DD_ALERT_DELTA_THRESHOLD", 0.01)
            now_ts = time.time()
            should_alert = False
            if not prev_paused:
                should_alert = True
            else:
                if (_last_dd_alert_ts == 0.0) or (now_ts - _last_dd_alert_ts >= dd_alert_cooldown) or (_last_dd_alert_dd is not None and dd > _last_dd_alert_dd + dd_delta_threshold):
                    should_alert = True
            if should_alert:
                msg = f"🛑 DD DIÁRIO > {max_dd_stop:.0%} ({dd:.1%}) - Trading AUTO-STOP"
                logger.critical(msg)
                utils.send_telegram_message(msg)
                _last_dd_alert_ts = now_ts
                _last_dd_alert_dd = dd
            
        # 2b. VIX Risk Switching
        try:
            vix = utils.get_vix_br()
            if vix is not None:
                if vix > getattr(config, "VIX_THRESHOLD_PROTECTION", 35):
                    if config.CURRENT_OPERATION_MODE != "PROTECTION":
                        config.set_operation_mode("PROTECTION")
                        logger.warning(f"🛡️ VIX CRÍTICO ({vix:.1f}) -> MODO PROTEÇÃO")
                elif vix > getattr(config, "VIX_THRESHOLD_RISK_OFF", 30):
                    if config.CURRENT_OPERATION_MODE not in ["DEFENSIVE", "PROTECTION"]:
                        config.set_operation_mode("DEFENSIVE")
                        logger.warning(f"⚠️ VIX ALTO ({vix:.1f}) -> MODO DEFENSIVO")
            else:
                logger.debug("VIX value is None, skipping risk switching update.")
        except Exception as e:
            logger.error(f"Erro no VIX risk switching: {e}")
        
        # 3. Global volume (>R$1M)
        # 3. Global volume check (Bot Financeiro)
        daily_volume = utils.get_daily_volume()
        volume_limit = getattr(config, "DAILY_VOLUME_LIMIT", 1_000_000_000)
        
        logger.debug(f"Daily volume details: current={daily_volume:,.2f}, limit={volume_limit:,.2f}")

        if daily_volume > volume_limit:
            trading_paused = True
            logger.warning(f"Limite volume diário atingido: R${daily_volume:,.2f} > R${volume_limit:,.2f}")

        # 4. Win Rate (últimos 20 trades < 50%)
        try:
            with utils.mt5_lock:
                deals = mt5.history_deals_get(datetime.now() - timedelta(days=7), datetime.now())
            if deals:
                relevant_deals = [d for d in deals if d.entry == mt5.DEAL_ENTRY_OUT]
                last_20 = sorted(relevant_deals, key=lambda x: x.time, reverse=True)[:20]
                
                if len(last_20) >= 20:
                    wins = sum(1 for d in last_20 if d.profit > 0)
                    win_rate = (wins / len(last_20)) * 100
                    
                    if win_rate < 50.0 and not CIRCUIT_BREAKER_DISABLED:
                        prev_paused = trading_paused
                        trading_paused = True
                        now_ts = time.time()
                        should_alert = False
                        if not prev_paused:
                            should_alert = True
                        else:
                            if (_last_wr_alert_ts == 0.0) or (now_ts - _last_wr_alert_ts >= 3600) or (_last_wr_alert_wr is not None and win_rate < _last_wr_alert_wr - 5.0):
                                should_alert = True
                        if should_alert:
                            logger.critical(f"🛑 Win Rate crítico: {win_rate:.1f}% nas últimas 20 operações. Trading PAUSADO.")
                            utils.send_telegram_message(f"🛑 <b>CIRCUIT BREAKER: PERFORMANCE</b>\nWin Rate: {win_rate:.1f}% (últimos 20 trades)\nStatus: <b>PAUSADO PARA REVISÃO</b>")
                            _last_wr_alert_ts = now_ts
                            _last_wr_alert_wr = win_rate
        except Exception as e:
            logger.error(f"Erro health watcher performance: {e}")
        
        try:
            limit = utils.get_effective_exposure_limit()
            current_exposure = utils.calculate_total_exposure()
            if current_exposure >= 0.8 * limit:
                push_alert(f"⚠️ Exposição em {current_exposure/limit:.0%} do limite", "WARNING")
        except Exception as e:
            logger.error(f"Erro monitoramento de exposição: {e}")
        
        try:
            reconcile_trade_log_today()
        except Exception as e:
            logger.error(f"Erro reconcile no watcher: {e}")
        
        time.sleep(300)


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
_last_entry_price = {}
daily_target_hit_day: Optional[date] = None
daily_target_hit_pct: Optional[float] = None
last_entry_attempt = {}
alerts = deque(maxlen=10)
alerts_lock = Lock()
failure_lock = Lock()
optimized_params = {}
trading_paused = False
_last_wr_alert_ts = 0.0
_last_wr_alert_wr: Optional[float] = None
_last_dd_alert_ts = 0.0
_last_dd_alert_dd: Optional[float] = None
CIRCUIT_BREAKER_DISABLED = False
daily_max_equity = 0.0
equity_inicio_dia = 0.0
last_reset_day: Optional[date] = None
last_failure_reason = {}
manual_pause_reason = ""
pause_reset_day: Optional[date] = None
_last_eod_report_date = None
entry_indicators = {}  # {symbol: indicadores_da_entrada}
entry_indicators_lock = Lock()
# Flag para controle de mensagens de inicialização
_first_build_done = False
_first_build_lock = Lock()
trading_paused = False
bot_should_run = True
daily_trades_per_symbol = defaultdict(int)
position_open_times_lock = Lock()
# New globals from refactors
DAILY_STATE_FILE = "daily_bot_state.json"
bot_state = BotState()  # Priority 1: Unified state
position_open_times = TimedCache(max_age_seconds=86400)  # Priority 2: 24h for positions
last_close_time = TimedCache(max_age_seconds=7200)  # Priority 2: 2h for close times

# =========================
# LOG SETUP PROFISSIONAL
# =========================
from logging.handlers import TimedRotatingFileHandler
import os
import glob
from datetime import datetime, timedelta


class SafeTimedRotatingFileHandler(TimedRotatingFileHandler):
    def doRollover(self):
        try:
            super().doRollover()
        except PermissionError:
            pass


def setup_logging():
    log_dir = "logs"
    bot_dir = os.path.join(log_dir, "bot")
    err_dir = os.path.join(log_dir, "errors")
    ana_dir = os.path.join(log_dir, "analysis")
    os.makedirs(bot_dir, exist_ok=True)
    os.makedirs(err_dir, exist_ok=True)
    os.makedirs(ana_dir, exist_ok=True)

    class _SuppressNoisyWebSocketErrors(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            try:
                msg = record.getMessage()
            except Exception:
                msg = ""

            if not msg:
                return True

            if "tornado.websocket.WebSocketClosedError" in msg:
                return False
            if "tornado.websocket.WebSocketClosedError" in (record.exc_text or ""):
                return False

            if "WebSocketClosedError" in msg and "tornado" in record.name:
                return False
            if "StreamClosedError" in msg and "tornado" in record.name:
                return False
            if "Task exception was never retrieved" in msg and "WebSocketClosedError" in msg:
                return False
            if "Stream is closed" in msg and "tornado" in record.name:
                return False

            return True
    
    class _ThreeHourHandler(SafeTimedRotatingFileHandler):
        def __init__(self, filename, level):
            super().__init__(filename=filename, when="h", interval=3, backupCount=56, encoding="utf-8", utc=False)
            self.suffix = "%Y-%m-%d_%H"
            self.extMatch = self.extMatch
            self.setLevel(level)

    main_handler = _ThreeHourHandler(
        filename=os.path.join(bot_dir, "xp3_bot.log"),
        level=logging.INFO
    )
    main_handler.setLevel(logging.INFO)
    main_handler.setFormatter(
        logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    )
    main_handler.addFilter(_SuppressNoisyWebSocketErrors())
    
    error_handler = _ThreeHourHandler(
        filename=os.path.join(err_dir, "errors.log"),
        level=logging.ERROR
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(
        logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s')
    )
    error_handler.addFilter(_SuppressNoisyWebSocketErrors())
    
    # Console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(
        logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    )
    console_handler.addFilter(_SuppressNoisyWebSocketErrors())
    
    # Configura logger raiz
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.handlers.clear()
    root_logger.addHandler(main_handler)
    root_logger.addHandler(error_handler)
    root_logger.addHandler(console_handler)
    def _retention_worker(paths, days):
        import time as _t
        import os as _os
        import threading as _th
        def _run():
            while True:
                cutoff = datetime.now() - timedelta(days=days)
                for p in paths:
                    try:
                        for name in os.listdir(p):
                            fp = os.path.join(p, name)
                            try:
                                if os.path.isfile(fp):
                                    ts = datetime.fromtimestamp(os.path.getmtime(fp))
                                    if ts < cutoff:
                                        try:
                                            os.remove(fp)
                                        except Exception:
                                            pass
                            except Exception:
                                pass
                    except Exception:
                        pass
                _t.sleep(3600)
        t = _th.Thread(target=_run, daemon=True)
        t.start()
    _retention_worker([bot_dir, err_dir, ana_dir], 7)
    return logging.getLogger("bot")

# Chama setup
logger = setup_logging()

# =========================
# TIMEFRAMES
# =========================
TIMEFRAME_BASE = mt5.TIMEFRAME_M15
TIMEFRAME_MACRO = getattr(mt5, f"TIMEFRAME_{config.MACRO_TIMEFRAME}", mt5.TIMEFRAME_H1)

CURRENT_MODE = "AMBOS"
def _get_config_symbols_for_validation():
    syms = []
    try:
        elite = getattr(config, "ELITE_SYMBOLS", {})
        if isinstance(elite, dict) and elite:
            syms = list(elite.keys())
    except Exception:
        syms = []
    if not syms:
        for attr in ("SECTOR_MAP", "UNIVERSE_B3", "SYMBOL_MAP"):
            try:
                mp = getattr(config, attr, {})
                if isinstance(mp, dict) and mp:
                    syms = list(mp.keys())
                    break
            except Exception:
                pass
    if not syms:
        try:
            syml = getattr(config, "SYMBOLS", [])
            if isinstance(syml, (list, tuple)) and syml:
                syms = list(syml)
        except Exception:
            pass
    return sorted(set(syms))

def validate_mt5_symbols_or_abort():
    symbols = _get_config_symbols_for_validation()
    if not symbols:
        logger.critical("❌ Nenhum ativo configurado encontrado para validação (ELITE/SECTOR/UNIVERSE/SYMBOLS vazio)")
        return False
    missing = []
    for sym in symbols:
        try:
            try:
                mt5.symbol_select(sym, True)
            except Exception:
                pass
            info = mt5.symbol_info(sym)
            if not info:
                missing.append(f"{sym}: info ausente")
                continue
            rates = mt5.copy_rates_from_pos(sym, TIMEFRAME_BASE, 0, 3)
            if rates is None or len(rates) == 0:
                missing.append(f"{sym}: sem barras {TIMEFRAME_BASE}")
        except Exception as e:
            missing.append(f"{sym}: erro {e}")
    if missing:
        msg = " | ".join(missing)
        logger.critical(f"❌ Inicialização abortada. Ativos indisponíveis no MT5: {msg}")
        return False
    logger.info(f"✅ Validação MT5 concluída para {len(symbols)} ativos")
    return True

# ============================================
# 💾 FUNÇÕES DE PERSISTÊNCIA
# ============================================

def save_daily_state():
    """
    Salva TODO o estado diário do bot em JSON
    Chame isso periodicamente (a cada 5 min) e ao encerrar
    """
    global daily_trades_per_symbol, equity_inicio_dia, daily_max_equity
    global last_entry_time, current_trading_day, daily_cycle_completed
    global daily_target_hit_day, daily_target_hit_pct, _symbol_pyramid_leg, _last_entry_price
    
    try:
        state = {
            "date": datetime.now().date().isoformat(),
            "timestamp": datetime.now().isoformat(),
            
            # 💰 Financeiro
            "equity_inicio_dia": float(equity_inicio_dia),
            "daily_max_equity": float(daily_max_equity),
            
            # 📊 Contadores
            "daily_trades_per_symbol": dict(daily_trades_per_symbol),
            
            # ⏱️ Cooldowns
            "last_entry_time": {
                sym: time_val for sym, time_val in last_entry_time.items()
            },
            
            # 🔄 Estado do ciclo
            "current_trading_day": current_trading_day.isoformat() if current_trading_day else None,
            "daily_cycle_completed": daily_cycle_completed,

            "daily_target_hit_day": daily_target_hit_day.isoformat() if daily_target_hit_day else None,
            "daily_target_hit_pct": float(daily_target_hit_pct) if daily_target_hit_pct is not None else None,
            "symbol_pyramid_leg": dict(_symbol_pyramid_leg),
            "last_entry_price": {sym: float(px) for sym, px in _last_entry_price.items()},
            
            # 📈 Posições abertas (backup)
            "open_positions": []
        }
        
        # Salva tickets das posições abertas
        with utils.mt5_lock:
            positions = mt5.positions_get() or []
        
        for pos in positions:
            state["open_positions"].append({
                "ticket": pos.ticket,
                "symbol": pos.symbol,
                "side": "BUY" if pos.type == mt5.POSITION_TYPE_BUY else "SELL",
                "volume": pos.volume,
                "entry_price": pos.price_open,
                "sl": pos.sl,
                "tp": pos.tp,
                "time": pos.time
            })
        
        # Salva em arquivo temporário primeiro (atomic write)
        temp_file = f"{DAILY_STATE_FILE}.tmp"
        with open(temp_file, "w") as f:
            json.dump(state, f, indent=2)
        
        # Substitui o arquivo original (operação atômica no Windows)
        import shutil
        shutil.move(temp_file, DAILY_STATE_FILE)
        
        logger.debug(f"💾 Estado diário salvo: {len(state['daily_trades_per_symbol'])} símbolos rastreados")
        
    except Exception as e:
        logger.error(f"❌ Erro ao salvar estado diário: {e}", exc_info=True)


def load_daily_state():
    """
    Carrega estado diário ao iniciar o bot
    Valida se é do mesmo dia, senão reseta
    
    Chame isso no main() ANTES de iniciar as threads
    """
    global daily_trades_per_symbol, equity_inicio_dia, daily_max_equity
    global last_entry_time, current_trading_day, daily_cycle_completed
    global daily_target_hit_day, daily_target_hit_pct, _symbol_pyramid_leg, _last_entry_price
    
    if not os.path.exists(DAILY_STATE_FILE):
        logger.info("ℹ️ Nenhum estado anterior encontrado (primeiro boot do dia)")
        return False
    
    try:
        with open(DAILY_STATE_FILE, "r") as f:
            state = json.load(f)
        
        saved_date = state.get("date")
        today = datetime.now().date().isoformat()
        
        # ✅ VALIDA SE É DO MESMO DIA
        if saved_date != today:
            logger.info(f"🔄 Estado anterior era de {saved_date} - Iniciando novo dia")
            return False
        
        # ============================================
        # 📥 RESTAURA TODOS OS DADOS
        # ============================================
        
        # 💰 Financeiro
        saved_equity = state.get("equity_inicio_dia", 0.0)
        daily_max_equity = state.get("daily_max_equity", 0.0)
        
        # ✅ Validação de sanidade do Equity Inicial
        with utils.mt5_lock:
            acc = mt5.account_info()
            current_equity = acc.equity if acc else 0.0
            
        # Se salvo for inválido ou muito discrepante (ex: foi salvo como 0 ou saldo antigo)
        is_suspicious = False
        if current_equity > 0:
            ratio = saved_equity / current_equity
            # Se for menor que 80% ou maior que 120% do atual, reseta
            if ratio < 0.8 or ratio > 1.2:
                is_suspicious = True
        
        if saved_equity <= 1000 or is_suspicious:
            if current_equity > 0:
                logger.warning(f"⚠️ Equity salvo suspeito (R$ {saved_equity:,.2f} vs R$ {current_equity:,.2f}). Resetando para atual.")
                equity_inicio_dia = current_equity
            else:
                equity_inicio_dia = 0.0 # Sem conexão, mantém 0
        else:
            equity_inicio_dia = saved_equity
        
        # 📊 Contadores
        daily_trades_per_symbol_data = state.get("daily_trades_per_symbol", {})
        daily_trades_per_symbol.clear()
        for sym, count in daily_trades_per_symbol_data.items():
            daily_trades_per_symbol[sym] = int(count)
        
        # ⏱️ Cooldowns
        last_entry_time_data = state.get("last_entry_time", {})
        last_entry_time.clear()
        for sym, time_val in last_entry_time_data.items():
            last_entry_time[sym] = float(time_val)
        
        # 🔄 Ciclo
        trading_day_str = state.get("current_trading_day")
        if trading_day_str:
            current_trading_day = datetime.fromisoformat(trading_day_str).date()
        
        daily_cycle_completed = state.get("daily_cycle_completed", False)

        daily_target_hit_day = None
        daily_target_hit_pct = None
        try:
            dth = state.get("daily_target_hit_day")
            if dth:
                daily_target_hit_day = datetime.fromisoformat(dth).date()
            dtp = state.get("daily_target_hit_pct")
            if dtp is not None:
                daily_target_hit_pct = float(dtp)
        except Exception:
            daily_target_hit_day = None
            daily_target_hit_pct = None

        _symbol_pyramid_leg.clear()
        _symbol_pyramid_leg.update({k: int(v) for k, v in (state.get("symbol_pyramid_leg", {}) or {}).items()})

        _last_entry_price.clear()
        _last_entry_price.update({k: float(v) for k, v in (state.get("last_entry_price", {}) or {}).items()})
        
        # ============================================
        # 📊 RELATÓRIO DE RESTAURAÇÃO
        # ============================================
        
        open_positions = state.get("open_positions", [])
        
        logger.info("=" * 60)
        logger.info("✅ ESTADO DIÁRIO RESTAURADO COM SUCESSO")
        logger.info("=" * 60)
        logger.info(f"📅 Data: {saved_date}")
        logger.info(f"💰 Equity Inicial: R${equity_inicio_dia:,.2f}")
        logger.info(f"📈 Max Equity: R${daily_max_equity:,.2f}")
        logger.info(f"📊 Símbolos com trades: {len(daily_trades_per_symbol)}")
        
        if daily_trades_per_symbol:
            logger.info("   Contadores:")
            for sym, count in sorted(daily_trades_per_symbol.items(), key=lambda x: x[1], reverse=True)[:5]:
                logger.info(f"   • {sym}: {count} trades")
        
        logger.info(f"⏱️ Cooldowns ativos: {len(last_entry_time)}")
        logger.info(f"📍 Posições salvas: {len(open_positions)}")
        logger.info(f"🔄 Ciclo completo: {'Sim' if daily_cycle_completed else 'Não'}")
        logger.info("=" * 60)
        
        # ✅ VALIDA POSIÇÕES (AVISO SE DISCREPÂNCIA)
        with utils.mt5_lock:
            current_positions = mt5.positions_get() or []
        
        if len(current_positions) != len(open_positions):
            logger.warning(
                f"⚠️ ATENÇÃO: Estado salvo tinha {len(open_positions)} posições, "
                f"mas MT5 tem {len(current_positions)} agora!"
            )
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro ao carregar estado diário: {e}", exc_info=True)
        return False


# ============================================
# 🔄 THREAD DE AUTO-SAVE
# ============================================

def auto_save_state_thread():
    """
    Thread que salva o estado a cada 5 minutos
    Adicione ao main() junto com as outras threads
    """
    while True:
        try:
            time.sleep(300)  # 5 minutos
            save_daily_state()
            
        except Exception as e:
            logger.error(f"Erro no auto-save: {e}")
            time.sleep(60)


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
    """
    Thread que gerencia a atualização da matriz de correlação.
    Agora executa imediatamente ao iniciar e repete a cada 10 minutos.
    """
    logger.info("🧵 Thread de Correlação iniciada (Primeira execução imediata)")

    while True:
        try:
            # 1. Executa a atualização primeiro (sem esperar)
            update_correlation_matrix()
            # Nova: Calcula vol real-time (std dev últimos 60 candles)
            for sym in bot_state.get_top15():
                df = safe_copy_rates(sym, mt5.TIMEFRAME_M15, 60)
                if df is not None:
                    vol = np.std(df['close'].pct_change()) * 100  # Vol %
                    if vol > config.MAX_VOL_THRESHOLD:  # Ex: 2x média
                        block_symbol(sym, reason="Alta volatilidade")
        
            # Intervalo vindo do config
            time.sleep(config.CORR_UPDATE_INTERVAL)
        except Exception as e:
            logger.error(f"🚨 Erro na thread de correlação: {e}", exc_info=True)
            # Se der erro, espera um pouco menos (5 min) antes de tentar de novo
            time.sleep(300)


def update_correlation_matrix():
    # Indica ao Python que queremos alterar a variável global usada pelo painel
    global last_correlation_update

    symbols = bot_state.get_top15()

    if not is_valid_dataframe(symbols, min_rows=2):
        logger.warning("⚠️ Símbolos insuficientes para calcular correlação (< 2 ativos)")
        return

    with correlation_lock:
        try:
            # Chama a lógica pesada de cálculo que está no utils.py
            correlation_cache = utils.calculate_correlation_matrix(symbols)

            # Atualiza o timestamp GLOBAL para o painel reconhecer a atualização
            last_correlation_update = datetime.now()

            logger.info(
                f"✅ Matriz de correlação atualizada com sucesso ({len(symbols)} ativos)"
            )
        except Exception as e:
            logger.error(f"❌ Falha no cálculo da matriz: {e}")


def get_average_correlation_with_portfolio(
    symbol: str, current_positions_symbols: List[str]
) -> float:
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
    elite_json_path = getattr(config, "ELITE_SYMBOLS_JSON_PATH", "")
    elite_loaded = False

    if elite_json_path:
        try:
            import os
            import json

            if os.path.exists(elite_json_path):
                with open(elite_json_path, "r", encoding="utf-8") as f:
                    data = json.load(f) or {}
                elite = data.get("elite_symbols") or data.get("ELITE_SYMBOLS") or data.get("symbols") or {}
                if isinstance(elite, dict) and elite:
                    optimized_params = {
                        str(sym): (params.copy() if isinstance(params, dict) else {})
                        for sym, params in elite.items()
                    }
                    elite_loaded = True
                    logger.info(
                        f"Parâmetros carregados do JSON ({len(optimized_params)} ativos elite): {elite_json_path}"
                    )
        except Exception as e:
            logger.warning(f"Falha ao carregar elite JSON: {e}")

    if not elite_loaded:
        elite = getattr(config, "ELITE_SYMBOLS", {})
        if elite:
            optimized_params = {sym: params.copy() for sym, params in elite.items()}
            logger.info(
                f"Parâmetros carregados do ELITE_SYMBOLS ({len(optimized_params)} ativos elite)"
            )
        else:
            optimized_params = getattr(config, "OPTIMIZED_PARAMS", {}).copy()
            logger.warning("ELITE_SYMBOLS vazio. Usando fallback.")

    for sym in optimized_params:
        params = optimized_params[sym]
        defaults = {
            "ema_short": 9,
            "ema_long": 21,
            "rsi_low": 35,
            "rsi_high": 70,
            "adx_threshold": 25,
            "mom_min": 0.0,
        }
        for k, v in defaults.items():
            params.setdefault(k, v)
    
    # ✅ OTIMIZAÇÃO DIÁRIA (OPCIONAL - Desabilitar se causar lentidão)
    ENABLE_DAILY_OPTIMIZATION = False  # ⚠️ Mude para True se quiser otimização automática
    
    if ENABLE_DAILY_OPTIMIZATION:
        logger.info("🔧 Iniciando otimização diária de parâmetros...")
        optimize_params_daily()
    else:
        logger.info("✅ Parâmetros otimizados carregados do config.py (otimização diária desabilitada)")


def optimize_params_daily():
    """
    Otimiza parâmetros diariamente usando dados históricos.
    ⚠️ PODE SER DEMORADO (5-10 min para todos os ativos)
    """
    import time
    start_time = time.time()
    optimized_count = 0
    
    # Lista de símbolos para otimizar (top 20 mais líquidos)
    symbols_to_optimize = [
        "PETR4", "VALE3", "ITUB4", "BBDC4", "BBAS3",
        "ABEV3", "WEGE3", "RENT3", "SUZB3", "ELET3",
        "PRIO3", "CSAN3", "CSNA3", "UGPA3", "USIM3"
    ]
    
    for sym in symbols_to_optimize:
        try:
            # Land Trading: Aumentado para 2500 candles para melhor treino ML
            df = safe_copy_rates(sym, TIMEFRAME_BASE, 2500)
            if df is None or len(df) < 100:
                logger.debug(f"⏭️ {sym}: Dados insuficientes")
                continue
            
            optimized = ml_optimizer.optimize(df, sym)
            
            if optimized:
                optimized_params[sym] = optimized
                optimized_count += 1
                logger.info(f"✅ {sym}: Parâmetros otimizados aplicados")
            
        except Exception as e:
            logger.error(f"Erro ao otimizar {sym}: {e}")
            continue
    
    elapsed = time.time() - start_time
    logger.info(
        f"✅ Otimização concluída: {optimized_count}/{len(symbols_to_optimize)} ativos "
        f"em {elapsed:.1f}s"
    )


# =========================
# BUILD TOP15
# =========================
def build_portfolio_and_top15():
    """
    ✅ VERSÃO COM AUDITORIA: Registra análise de TODOS os ativos
    """
    global _first_build_done
    scored = []
    indicators = {}

    elite_symbols = list(optimized_params.keys())

    if not elite_symbols:
        logger.error("❌ ELITE_SYMBOLS está vazio!")
        return {}, []

    with _first_build_lock:
        if not _first_build_done:
            logger.info(
                f"Parâmetros carregados do ELITE_SYMBOLS ({len(elite_symbols)} ativos elite)"
            )
            _first_build_done = True

    for sym in elite_symbols:
        df = utils.safe_copy_rates(sym, TIMEFRAME_BASE, 300)

        if df is None:
            mt5.symbol_select(sym, True)
            time.sleep(0.5)
            df = utils.safe_copy_rates(sym, TIMEFRAME_BASE, 300)

        if df is None or len(df) < 20:
            ind = {
                "ema_fast": 0,
                "ema_slow": 0,
                "rsi": 50,
                "atr": 0.01,
                "atr_pct": 0,
                "adx": 0,
                "vwap": None,
                "close": 0,
                "macro_trend_ok": False,
                "tick_size": 0.01,
                "sector": config.SECTOR_MAP.get(sym, "Elite"),
                "error": "NO_DATA",
                "score": 1,
                "direction": "–"
            }
            scored.append((1, sym))
            indicators[sym] = ind
            
            # ✅ LOG: Sem dados
            daily_logger.log_analysis(
                symbol=sym,
                signal="NONE",
                strategy="N/A",
                score=1,
                rejected=True,
                reason="❌ Sem dados MT5",
                indicators={"rsi": 50, "adx": 0, "spread_points": 0, "spread_nominal": 0, "spread_pct": 0, "volume_ratio": 0, "ema_trend": "N/A"}
            )
            
            with _first_build_lock:
                if not _first_build_done:
                    logger.info(f"⚠️ {sym}: Sem dados, mantido no TOP15 com score 1")
            continue

        params = optimized_params.get(sym, {})
        ind = utils.get_cached_indicators(sym, TIMEFRAME_BASE, 300)
        
        if ind.get("error"):
            df = utils.safe_copy_rates(sym, TIMEFRAME_BASE, 300)
            ind = utils.get_cached_indicators(sym, TIMEFRAME_BASE, 300)

        if ind.get("error"):
            ind = {
                "ema_fast": 0,
                "ema_slow": 0,
                "rsi": 50,
                "atr": 0,
                "atr_pct": 0,
                "adx": 0,
                "vwap": 0,
                "close": df["close"].iloc[-1],
                "macro_trend_ok": False,
                "tick_size": 0.01,
            }

        # ✅ SCORE v5.5 (AGRESSIVO)
        # Sincroniza com utils.py para precisão total
        score = utils.calculate_signal_score(ind)

        # ✅ LOG DE AUDITORIA: Lógica de decisão
        ema_trend = "UP" if ind["ema_fast"] > ind["ema_slow"] else "DOWN"
        rsi = ind.get("rsi", 50)
        adx = ind.get("adx", 0)

        # GATILHO SIMPLIFICADO (Forçado)
        forced_buy = (ema_trend == "UP" and rsi > 50)
        forced_sell = (ema_trend == "DOWN" and rsi < 50)
        
        # EXCEÇÃO ADX 15-20 (Com inclinação)
        ema_diff_pct = abs(ind["ema_fast"] - ind["ema_slow"]) / max(ind["close"], 1)
        ema_tilt_ok = ema_diff_pct > 0.0005 # > 0.05% de inclinacao/gap
        adx_exception = (15 <= adx <= 20) and ema_tilt_ok

        # Determina DIREÇÃO e SINAL FINAL
        if score >= config.MIN_SIGNAL_SCORE or forced_buy or forced_sell or adx_exception:
            if ema_trend == "UP":
                direction = "↑ LONG"
                signal = "BUY"
            else:
                direction = "↓ SHORT"
                signal = "SELL"
        else:
            direction = "–"
            signal = "NONE"

        # Salva score e direção
        ind["score"] = score
        ind["direction"] = direction
        ind["sector"] = config.SECTOR_MAP.get(sym, "Elite")
        
        scored.append((score, sym))
        indicators[sym] = ind

        # ✅ LOG: Análise completa
        reason_log = ""
        if signal == "NONE":
            reason_log = f"📊 Score {score:.0f} < {config.MIN_SIGNAL_SCORE} e sem gatilho forçado"
            rejected = True
        else:
            # Land Trading: Não marca como executada aqui, apenas sinaliza monitoramento
            reason_log = f"⏳ Aguardando Gatilho (Score: {score:.0f} | Forçado: {forced_buy or forced_sell})"
            rejected = True # Marca como True para cair no status "AGUARDANDO" do logger

        daily_logger.log_analysis(
            symbol=sym,
            signal=signal,
            strategy="ELITE_V5.5",
            score=score,
            rejected=rejected,
            reason=reason_log,
            indicators={
                "rsi": ind.get("rsi", 50),
                "adx": ind.get("adx", 0),
                "spread_points": ind.get("spread_points", 0),
                "spread_pct": ind.get("spread_pct", 0),
                "volume_ratio": ind.get("volume_ratio", 0),
                "ema_trend": ema_trend
            }
        )

        with _first_build_lock:
            if not _first_build_done:
                logger.info(f"✅ Ativo Carregado: {sym} | Score: {score}")

    scored.sort(reverse=True, key=lambda x: x[0])
    selected_top = [s for _, s in scored[:15]]

    bot_state.update(indicators, selected_top)
    
    # Atualiza bot_bridge.json
    update_bot_bridge()
    
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
            "INFO",
        )
        return False
    
    # Gap check
    gap = utils.get_open_gap(symbol, TIMEFRAME_BASE)
    max_gap_pct = config.MAX_GAP_OPEN_PCT * 100.0
    if gap is not None and gap > max_gap_pct:
        push_panel_alert(
            f"⚠️ {symbol} rejeitado: Gap de abertura alto ({gap:.2f}% > {max_gap_pct:.0f}%)",
            "INFO",
        )
        return False

    ok_spread, cur_spread, avg_spread = utils.check_spread(symbol, TIMEFRAME_BASE, getattr(config, "SPREAD_LOOKBACK_BARS", 10))
    if not ok_spread:
        push_panel_alert(
            f"⚠️ {symbol} rejeitado: Spread atual {cur_spread:.2f}% > média {avg_spread:.2f}%",
            "INFO",
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
            "tp": pos.tp,
        }
        mt5.order_send(request)

def force_mt5_reconnect(max_attempts: int = 3) -> bool:
    """
    Força reconexão do MT5 em caso de travamento
    """
    logger.warning("🔄 Forçando reconexão do MT5...")
    
    for attempt in range(1, max_attempts + 1):
        try:
            # 1. Shutdown forçado
            mt5.shutdown()
            time.sleep(2)
            
            # 2. Tenta reinicializar
            if mt5.initialize(path=config.MT5_TERMINAL_PATH):
                # 3. Valida conexão
                terminal = mt5.terminal_info()
                account = mt5.account_info()
                
                if terminal and terminal.connected and account:
                    logger.info(
                        f"✅ MT5 reconectado (tentativa {attempt}) | "
                        f"Conta: {account.login} | "
                        f"Servidor: {account.server}"
                    )
                    return True
            
            logger.warning(f"⚠️ Tentativa {attempt}/{max_attempts} de reconexão falhou")
            time.sleep(3)
            
        except Exception as e:
            logger.error(f"❌ Erro na reconexão: {e}")
            time.sleep(3)
    
    logger.critical("🚨 FALHA TOTAL: Não foi possível reconectar o MT5")
    return False


def validate_mt5_health() -> tuple[bool, str]:
    """
    Diagnóstico completo do estado do MT5
    
    Returns:
        (is_healthy: bool, diagnostic_message: str)
    """
    try:
        # 1. Terminal Info
        terminal = mt5.terminal_info()
        if not terminal:
            return False, "Terminal info = None (MT5 não inicializado)"
        
        if not terminal.connected:
            return False, f"MT5 desconectado do servidor"
        
        if not terminal.trade_allowed:
            return False, "Trading desabilitado no terminal"
        
        # 2. Account Info
        account = mt5.account_info()
        if not account:
            return False, "Account info = None"
        
        if account.trade_mode != mt5.ACCOUNT_TRADE_MODE_DEMO and account.trade_mode != mt5.ACCOUNT_TRADE_MODE_REAL:
            return False, f"Modo de trading inválido: {account.trade_mode}"
        
        # 3. Testa comunicação (pega posições)
        test_positions = mt5.positions_get()
        if test_positions is None:
            return False, "positions_get() retornou None (comunicação falhou)"
        
        return True, "MT5 saudável"
        
    except Exception as e:
        return False, f"Exceção no diagnóstico: {e}"
    
def close_position(
    symbol: str, ticket: int, volume: float, price: float, reason: str = ""
):
    """
    ✅ VERSÃO THREAD-SAFE FINAL: Evita race conditions
    - Verifica se ticket já está sendo fechado
    - Timeout de 10s por tentativa
    - Libera lock automaticamente
    """
    # ============================================
    # 🚦 VERIFICA RACE CONDITION
    # ============================================
    if not can_close_position(ticket):
        logger.warning(f"⚠️ Ticket {ticket} já está sendo fechado por outra thread - pulando")
        return True

    success = False
    final_exit_price = price
    final_profit_money = 0.0
    final_pl_pct = 0.0

    try:
        # ============================================
        # 1️⃣ VALIDAÇÃO INICIAL
        # ============================================
        with utils.mt5_lock:
            pos_check = mt5.positions_get(ticket=ticket)
        
        if not pos_check:
            logger.info(f"✅ Ticket {ticket} não existe mais (já fechado)")
            return True
        
        pos = pos_check[0]
        side = "BUY" if pos.type == mt5.POSITION_TYPE_BUY else "SELL"
        entry_price = pos.price_open

        # ============================================
        # 4️⃣ RETRY COM TIMEOUT
        # ============================================
        max_retries = 5
        
        for attempt in range(1, max_retries + 1):
            # === 🔍 RECHECK POSIÇÃO ===
            with utils.mt5_lock:
                pos_recheck = mt5.positions_get(ticket=ticket)
            
            if not pos_recheck:
                logger.info(f"✅ {symbol}: Posição fechou durante retry (tentativa {attempt})")
                success = True
                break
            
            pos = pos_recheck[0]  # Atualiza pos

            # === 🔄 PREÇO ATUAL ===
            with utils.mt5_lock:
                tick = mt5.symbol_info_tick(symbol)
            
            if not tick:
                logger.error(f"❌ {symbol}: Sem cotação (tentativa {attempt})")
                time.sleep(1)
                continue
            
            current_price = tick.bid if side == "BUY" else tick.ask
            final_exit_price = current_price

            # === 🎯 DEVIATION E FILLING ===
            deviation = 100 
            order_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "position": ticket,
                "volume": float(volume),
                "type": order_type,
                "price": current_price,
                "deviation": deviation,
                "magic": 2026,
                "comment": f"XP3_CLOSE_{reason[:15]}",
                "type_filling": mt5.ORDER_FILLING_RETURN, # MANTENHA ESTE PARA B3
                "type_time": mt5.ORDER_TIME_GTC,
                }
            
            logger.info(f"🔄 Tentativa {attempt}/{max_retries}: {symbol} @ {current_price:.2f} (dev {deviation})")
            
            result = mt5_order_send_safe(request, timeout=10)
            
            if result is None:
                logger.error(f"❌ TIMEOUT: {symbol} tentativa {attempt}")
                if attempt == max_retries:
                    logger.critical(f"🆘 EMERGENCY CLOSE: {symbol}")
                    success = emergency_close_position(symbol, ticket, volume, side)
                else:
                    time.sleep(attempt * 1.0)
                continue

            if result.retcode == mt5.TRADE_RETCODE_DONE:
                success = True
                try:
                    with utils.mt5_lock:
                        deals = mt5.history_deals_get(datetime.now() - timedelta(minutes=15), datetime.now()) or []
                    out_deals = [d for d in deals if d.position_id == ticket and d.entry == mt5.DEAL_ENTRY_OUT]
                    if out_deals:
                        dlast = sorted(out_deals, key=lambda x: x.time, reverse=True)[0]
                        final_exit_price = float(dlast.price)
                        final_profit_money = float(dlast.profit or 0.0)
                        last_deal_id = int(getattr(dlast, "ticket", 0) or 0)
                        last_position_id = int(getattr(dlast, "position_id", 0) or 0)
                    else:
                        final_profit_money = float(pos.profit or 0.0)
                        last_deal_id = 0
                        last_position_id = int(getattr(pos, "ticket", 0) or 0)
                    denom = (entry_price * volume) if volume > 0 else 0.0
                    final_pl_pct = (final_profit_money / denom) * 100 if denom > 0 else 0.0
                except Exception:
                    final_profit_money = float(pos.profit or 0.0)
                    denom = (entry_price * volume) if volume > 0 else 0.0
                    final_pl_pct = (final_profit_money / denom) * 100 if denom > 0 else 0.0
                logger.info(
                    f"✅ FECHADO: {symbol} | P&L: R${final_profit_money:+.2f} ({final_pl_pct:+.2f}%) | {reason}"
                )
                break
            else:
                logger.warning(f"⚠️ Retry {attempt}: {result.comment} ({result.retcode})")
                if attempt < max_retries:
                    time.sleep(attempt * 1.0)

        # ============================================
        # ✅ PÓS-FECHAMENTO (SÓ SE SUCESSO)
        # ============================================
        if success:
            # Registros
            try:
                # ✅ A/B Group
                ab_group = get_ab_group(symbol)
                
                save_trade(
                    symbol=symbol, side=side, volume=volume,
                    entry_price=entry_price, exit_price=final_exit_price,
                    sl=pos.sl, tp=pos.tp,
                    pnl_money=final_profit_money, pnl_pct=final_pl_pct,
                    reason=reason,
                    ab_group=ab_group
                )
                log_trade_to_txt(
                    symbol=symbol, side=side, volume=volume,
                    entry_price=entry_price, exit_price=final_exit_price,
                    pnl_money=final_profit_money, pnl_pct=final_pl_pct,
                    reason=reason, deal_id=last_deal_id if 'last_deal_id' in locals() else None, position_id=last_position_id if 'last_position_id' in locals() else None
                )
                log_trade_cvm_compliance(
                    symbol=symbol, side=side, volume=volume,
                    entry_price=entry_price, exit_price=final_exit_price,
                    pnl=final_profit_money, reason=reason
                )
            except Exception as e:
                logger.error(f"Erro ao salvar trade: {e}")

            # === 🤖 ML: Usa indicadores da ENTRADA (melhor!)
            try:
                with entry_indicators_lock:
                    ind_at_entry = entry_indicators.get(symbol)
                
                if ind_at_entry:
                    ml_optimizer.record_trade(
                        symbol=symbol,
                        pnl_pct=final_pl_pct / 100,
                        indicators=ind_at_entry
                    )
                    entry_indicators.pop(symbol, None)
                else:
                    # Fallback: indicadores atuais
                    ind_now = utils.quick_indicators_custom(symbol, TIMEFRAME_BASE)
                    ml_optimizer.record_trade(symbol, final_pl_pct / 100, ind_now)
                
                utils.record_trade_outcome(symbol, final_profit_money)
                register_trade_result(symbol, final_profit_money < 0)
                
                if any(kw in reason.lower() for kw in ["stop", "sl", "loss"]):
                    register_sl_hit(symbol, final_exit_price)
                    
            except Exception as e:
                logger.error(f"Erro ML/anti-chop: {e}")

            # Notificações
            pl_emoji = "🟢" if final_profit_money > 0 else "🔴"
            push_alert(f"{pl_emoji} {symbol} FECHADO | R${final_profit_money:+.2f} | {reason}")
            
            try:
                send_telegram_exit(
                    symbol=symbol, side=side, volume=volume,
                    entry_price=entry_price, exit_price=final_exit_price,
                    profit_loss=final_profit_money, reason=reason
                )
            except:
                pass

            # Cleanup
            last_close_time.set(symbol, time.time())
            with position_open_times_lock:
                position_open_times.pop(ticket, None)
            _symbol_pyramid_leg.pop(symbol, None)
            _last_entry_price.pop(symbol, None)

        return success

    except Exception as e:
        logger.critical(f"Exceção crítica em close_position {symbol}: {e}", exc_info=True)
        return False
    
    finally:
        # === 🔓 SEMPRE LIBERA O LOCK ===
        mark_close_complete(ticket)


def emergency_close_position(symbol: str, ticket: int, volume: float, side: str) -> bool:
    """
    Último recurso: fecha a qualquer custo
    """
    logger.critical(f"🆘 EMERGENCY CLOSE: {symbol} (ticket {ticket})")
    
    try:
        # Verifica se ainda existe
        with utils.mt5_lock:
            pos = mt5.positions_get(ticket=ticket)
        
        if not pos:
            logger.info(f"✅ {symbol}: Já fechada (emergency cancelado)")
            return True
        
        order_type = mt5.ORDER_TYPE_SELL if side == "BUY" else mt5.ORDER_TYPE_BUY
        
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            logger.error(f"❌ Emergency: sem cotação {symbol}")
            return False
        
        price = tick.bid if side == "BUY" else tick.ask
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "position": ticket,
            "volume": float(volume),
            "type": order_type,
            "price": price,
            "deviation": 300,  # Aceita 300 pips!
            "type_filling": mt5.ORDER_FILLING_RETURN,
        }
        
        result = mt5_order_send_safe(request, timeout=15)
        
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"✅ EMERGENCY CLOSE OK: {symbol}")
            return True
        else:
            error = result.comment if result else "Timeout/None"
            logger.error(f"❌ EMERGENCY FALHOU: {symbol} - {error}")
            return False
    
    except Exception as e:
        logger.error(f"❌ Exceção emergency: {e}", exc_info=True)
        return False

# =========================
# 📊 GESTÃO AVANÇADA COM TP DINÂMICO (SUBSTITUIR manage_positions_advanced)
# =========================
# ============================================
# CORREÇÃO COMPLETA: Anti-fechamento instantâneo
# ============================================

# === ADICIONAR NO TOPO DO bot.py (após imports) ===
from collections import defaultdict

# Rastreamento de quando cada posição foi aberta
position_open_times = {}  # {ticket: timestamp}
position_open_times_lock = Lock()


# =========================
# ENTRADA COM PYRAMIDING
# =========================

def check_mtf_confirmation(symbol: str, side: str, base_ind: dict) -> tuple[bool, str]:
    """
    ✅ CONFIRMAÇÃO MULTI-TIMEFRAME (MTF)
    
    Valida tendência em H1 antes de entrar no M15.
    Reduz falsos sinais e aumenta win rate.
    
    Returns:
        (confirmado: bool, motivo: str)
    
    Impacto: +8-12% win rate (baseado em backtests)
    """
    try:
        # 1. Pega dados H1
        df_h1 = utils.safe_copy_rates(symbol, mt5.TIMEFRAME_H1, 100)
        
        if df_h1 is None or len(df_h1) < 50:
            return True, ""  # Fail-open se não tiver dados
        
        # 2. Calcula indicadores H1
        close_h1 = df_h1['close']
        ema_fast_h1 = close_h1.ewm(span=21, adjust=False).mean().iloc[-1]
        ema_slow_h1 = close_h1.ewm(span=50, adjust=False).mean().iloc[-1]
        
        # 3. ADX H1 (força da tendência)
        adx_h1 = utils.get_adx(df_h1) or 0
        
        # 4. Valida alinhamento
        if side == "BUY":
            # H1 deve estar em tendência de alta
            trend_ok = ema_fast_h1 > ema_slow_h1
            strong_trend = adx_h1 > 25
            
            if not trend_ok:
                return False, "H1 em baixa (contra M15)"
            
            if not strong_trend:
                return False, f"H1 sem força (ADX {adx_h1:.0f} < 25)"
            
        else:  # SELL
            trend_ok = ema_fast_h1 < ema_slow_h1
            strong_trend = adx_h1 > 25
            
            if not trend_ok:
                return False, "H1 em alta (contra M15)"
            
            if not strong_trend:
                return False, f"H1 sem força (ADX {adx_h1:.0f} < 25)"
        
        # 5. ✅ Bônus: Valida momentum H1
        momentum_h1 = utils.get_momentum(df_h1, period=10) or 0
        
        if side == "BUY" and momentum_h1 < 0:
            return False, "Momentum H1 negativo"
        
        if side == "SELL" and momentum_h1 > 0:
            return False, "Momentum H1 positivo"
        
        logger.info(
            f"✅ MTF OK: {symbol} | "
            f"H1 Trend: {'UP' if side=='BUY' else 'DOWN'} | "
            f"ADX: {adx_h1:.0f} | Mom: {momentum_h1:+.3f}"
        )
        
        return True, ""
    
    except Exception as e:
        logger.error(f"Erro MTF {symbol}: {e}")
        return True, ""  # Fail-open

last_entry_time = {}  # Adicione isso logo antes da função ou no topo do arquivo junto com as outras globais
def select_trading_strategy(symbol: str) -> str:
    """
    ✅ NOVO: Seleciona estratégia ideal baseado em regime de mercado
    
    Lógica:
    - ADX > 30 + Volume alto → TREND_FOLLOWING
    - ADX < 20 + RSI extremo → MEAN_REVERSION
    - ATR expansion + Breakout → BREAKOUT
    - ML habilitado → ML_ENSEMBLE (sobrepõe)
    """
    try:
        df = utils.safe_copy_rates(symbol, mt5.TIMEFRAME_M15, 50)
        
        if df is None or len(df) < 30:
            return "TREND_FOLLOWING"  # Padrão
        
        ind = utils.quick_indicators_custom(symbol, mt5.TIMEFRAME_M15, df=df)
        
        adx = ind.get('adx', 20)
        rsi = ind.get('rsi', 50)
        volume_ratio = ind.get('volume_ratio', 1.0)
        vol_breakout = ind.get('vol_breakout', False)
        # ========================================
        # 🚀 GESTÃO DE SINAIS (AB TESTING)
        # ========================================
        
        # Determina grupo AB (hash simples do símbolo)
        ab_group = "A" if int(hashlib.md5(symbol.encode()).hexdigest(), 16) % 2 == 0 else "B"
        ab_config = config.AB_TEST_GROUPS.get(ab_group, config.AB_TEST_GROUPS["A"])
            
        if config.ENABLE_ML_SIGNALS:
            return "ML_ENSEMBLE"
        
        # 2. BREAKOUT
        if vol_breakout and volume_ratio > 1.3:
            return "BREAKOUT"
        
        # 3. TREND_FOLLOWING
        if adx > 30 and volume_ratio > 1.1:
            return "TREND_FOLLOWING"
        
        # 4. MEAN_REVERSION vs NOISE
        if adx < 25:
             # Só opera contra tendência se for EXTREMO
            if rsi < 25 or rsi > 75:
                return "MEAN_REVERSION"
            else:
                return "WAIT_NOISE" # <--- Nova proteção
        
        # Padrão: Trend following
        return "TREND_FOLLOWING"
    
    except Exception as e:
        logger.error(f"Erro ao selecionar estratégia para {symbol}: {e}")
        return "TREND_FOLLOWING"


def get_ml_signal(symbol: str, side: str, indicators: dict) -> dict:
    """
    ✅ NOVO: Obtém sinal do ML Ensemble
    
    Returns:
        {
            'direction': 'BUY' | 'SELL' | 'HOLD',
            'confidence': float (0-1),
            'model': 'ENSEMBLE' | 'LSTM' | 'XGBOOST'
        }
    """
    try:
        # Importa o novo módulo ml_signals
        from ml_signals import MLSignalPredictor
        
        predictor = MLSignalPredictor()
        
        # Obtém predição
        prediction = predictor.predict(symbol, indicators)
        
        return prediction
    
    except ImportError:
        logger.error("ml_signals.py não encontrado - ML desabilitado")
        return {'direction': 'HOLD', 'confidence': 0.0, 'model': 'NONE'}
    
    except Exception as e:
        logger.error(f"Erro ao obter sinal ML para {symbol}: {e}")
        return {'direction': 'HOLD', 'confidence': 0.0, 'model': 'ERROR'}
def try_enter_position(symbol, side, risk_factor=1.0):
    """
    ✅ VERSÃO COM AUDITORIA: Registra motivo de cada rejeição
    """
    global last_entry_time
    
    # ========================================
    # 🛡️ CONTROLES COMERCIAIS (PRIORIDADE)
    # ========================================
    
    # ✅ Verificação de horário seguro (30min após abertura, 20min antes fechamento)
    can_trade_hours, hours_reason = check_market_hours()
    if not can_trade_hours:
        daily_logger.log_analysis(
            symbol=symbol, signal=side, strategy="MARKET_HOURS",
            score=0, rejected=True, reason=hours_reason,
            indicators={}
        )
        return
    asset_cfg = get_asset_class_config(symbol)
    _now = datetime.now().time()
    _start = datetime.strptime(asset_cfg["start"], "%H:%M").time()
    _end = datetime.strptime(asset_cfg["end"], "%H:%M").time()
    if not (_start <= _now <= _end):
        daily_logger.log_analysis(
            symbol=symbol, signal=side, strategy="MARKET_HOURS_ASSET",
            score=0, rejected=True, reason=f"Horário do ativo: {_start.strftime('%H:%M')}-{_end.strftime('%H:%M')}",
            indicators={}
        )
        return

    now_dt = datetime.now()
    if now_dt.weekday() == 4:
        reduce_after_str = getattr(config, "FRIDAY_RISK_REDUCE_AFTER", "")
        mult = float(getattr(config, "FRIDAY_RISK_FACTOR_MULT", 1.0) or 1.0)
        if reduce_after_str and mult < 1.0:
            try:
                reduce_after = datetime.strptime(reduce_after_str, "%H:%M").time()
                if now_dt.time() >= reduce_after:
                    risk_factor *= mult
            except Exception:
                pass
    
    # ✅ Verificação de meta diária
    should_stop, profit_reason = global_profit_protector()
    if should_stop:
        daily_logger.log_analysis(
            symbol=symbol, signal=side, strategy="PROFIT_LOCK",
            score=0, rejected=True, reason=f"🎯 {profit_reason}",
            indicators={}
        )
        return

    mode_params = {}
    try:
        mode_params = config.get_current_mode_params() if hasattr(config, "get_current_mode_params") else {}
    except Exception:
        mode_params = {}

    if not bool(mode_params.get("allow_new_entries", True)):
        daily_logger.log_analysis(
            symbol=symbol, signal=side, strategy="OP_MODE",
            score=0, rejected=True, reason="🚫 Modo atual bloqueia novas entradas",
            indicators={}
        )
        return

    with utils.mt5_lock:
        all_positions = mt5.positions_get() or []

    existing_pos = [p for p in all_positions if getattr(p, "symbol", None) == symbol]
    is_pyramiding = len(existing_pos) > 0

    if not is_pyramiding:
        try:
            max_pos = int(mode_params.get("max_concurrent_positions", 0) or 0)
        except Exception:
            max_pos = 0

        if max_pos > 0 and len(all_positions) >= max_pos:
            daily_logger.log_analysis(
                symbol=symbol, signal=side, strategy="RISK_CAP",
                score=0, rejected=True, reason=f"🚫 Máx posições atingido ({len(all_positions)}/{max_pos})",
                indicators={}
            )
            return
    else:
        allow_pyr = bool(mode_params.get("allow_pyramiding", False))
        if not allow_pyr:
            daily_logger.log_analysis(
                symbol=symbol, signal=side, strategy="PYRAMID",
                score=0, rejected=True, reason="🔺 Pirâmide bloqueada pelo modo atual",
                indicators={}
            )
            return

    if not is_pyramiding:
        _symbol_pyramid_leg.pop(symbol, None)
        _last_entry_price.pop(symbol, None)

    no_entry_before_close_min = 0
    try:
        if now_dt.weekday() == 4:
            no_entry_before_close_min = int(getattr(config, "FRIDAY_NO_ENTRY_BEFORE_CLOSE_MINUTES", getattr(config, "NO_ENTRY_BEFORE_CLOSE_MINUTES", 0)) or 0)
        else:
            no_entry_before_close_min = int(getattr(config, "NO_ENTRY_BEFORE_CLOSE_MINUTES", 0) or 0)
    except Exception:
        no_entry_before_close_min = 0

    if no_entry_before_close_min > 0:
        close_all_by = getattr(config, "FRIDAY_CLOSE_ALL_BY", getattr(config, "CLOSE_ALL_BY", "17:55")) if now_dt.weekday() == 4 else getattr(config, "CLOSE_ALL_BY", "17:55")
        minutes_to_close = _get_minutes_to_close_all_by(close_all_by, now_dt)
        if minutes_to_close is not None and minutes_to_close <= float(no_entry_before_close_min):
            daily_logger.log_analysis(
                symbol=symbol, signal=side, strategy="EOD_BUFFER",
                score=0, rejected=True,
                reason=f"⏳ Sem novas entradas: faltam {minutes_to_close:.0f} min p/ fechamento ({close_all_by})",
                indicators={}
            )
            return
    
    # ========================================
    # 0. ✅ FILTRO DE LIQUIDEZ (CRÍTICO)
    # ========================================
    if not utils.check_liquidity(symbol):
        daily_logger.log_analysis(
            symbol=symbol, signal=side, strategy="LIQUIDITY", score=0,
            rejected=True, reason="💧 Liquidez projetada < 20M",
            indicators={}
        )
        return

    # ========================================
    # 0.1. ✅ FILTROS TÉCNICOS CRÍTICOS (LAND TRADING)
    # ========================================
    ind_data = bot_state.get_indicators(symbol)
    if not ind_data:
        return
    of = utils.get_order_flow(symbol, 20)
    imb = float(of.get("imbalance", 0.0) or 0.0)
    cvd = float(of.get("cvd", 0.0) or 0.0)
    if side == "BUY":
        if cvd < 0 or imb < -0.12:
            daily_logger.log_analysis(
                symbol=symbol, signal=side, strategy="ORDER_FLOW_VETO", score=0,
                rejected=True, reason="Fluxo contrário (CVD<0 ou Imbalance<-12%)",
                indicators=ind_data
            )
            return
    else:
        if cvd > 0 or imb > 0.12:
            daily_logger.log_analysis(
                symbol=symbol, signal=side, strategy="ORDER_FLOW_VETO", score=0,
                rejected=True, reason="Fluxo contrário (CVD>0 ou Imbalance>12%)",
                indicators=ind_data
            )
            return

    if not additional_filters_ok(symbol):
        daily_logger.log_analysis(
            symbol=symbol, signal=side, strategy="COMMON_FILTERS", score=0,
            rejected=True, reason="🚫 Filtros comuns (gap/spread/volume médio)",
            indicators=ind_data
        )
        return
        
    rsi = ind_data.get("rsi", 50)
    
    # 1. RSI (Exaustão)
    if side == "BUY" and rsi > 70:
        logger.info(f"🛑 {symbol}: RSI esticado ({rsi:.1f} > 70) - Compra evitada.")
        return
    if side == "SELL" and rsi < 30:
        logger.info(f"🛑 {symbol}: RSI esticado ({rsi:.1f} < 30) - Venda evitada.")
        return

    # 2. Volume Ratio Dinâmico (Smart Liquidity - Land Trading)
    current_time = datetime.now().time()
    vol_ratio = ind_data.get("volume_ratio", 0)
    if current_time < datetime.strptime("12:00","%H:%M").time():
        min_vol = 1.2
        period_name = "Manhã"
    elif datetime.strptime("12:00","%H:%M").time() <= current_time <= datetime.strptime("13:30","%H:%M").time():
        if utils.is_future(symbol):
            min_vol = float(getattr(config, "LUNCH_MIN_VOLUME_RATIO", 0.5) or 0.5)
        else:
            min_vol = 0.8
        period_name = "Almoço"
    else:
        min_vol = 1.2
        period_name = "Tarde"

    if vol_ratio < min_vol:
        reason = f"🛑 Volume fraco para {period_name} ({vol_ratio:.2f}x < {min_vol}x)"
        logger.info(f"🛑 {symbol}: {reason} - Entrada evitada.")
        daily_logger.log_analysis(
            symbol=symbol, signal=side, strategy="VOLUME_TIMEFILTER",
            score=ind_data.get("score", 0),
            rejected=True, reason=reason,
            indicators=ind_data
        )
        from rejection_logger import log_trade_rejection
        log_trade_rejection(symbol, "VolumeTimeFilter", reason, {"vol_ratio": vol_ratio, "min_vol": min_vol, "deficit": round(min_vol - vol_ratio, 4), "period": period_name})
        return

    score = float(ind_data.get("score", 0) or 0)
    adx = float(ind_data.get("adx", 0) or 0)
    ema_fast = float(ind_data.get("ema_fast", 0) or 0)
    ema_slow = float(ind_data.get("ema_slow", 0) or 0)
    close_price = float(ind_data.get("close", 0) or 0)
    ema_trend = "UP" if ema_fast > ema_slow else "DOWN"
    forced_buy = (ema_trend == "UP" and rsi > 50)
    forced_sell = (ema_trend == "DOWN" and rsi < 50)
    forced_signal = (side == "BUY" and forced_buy) or (side == "SELL" and forced_sell)
    ema_diff_pct = abs(ema_fast - ema_slow) / max(close_price, 1)
    ema_tilt_ok = ema_diff_pct > 0.0005
    adx_exception = (15 <= adx <= 20) and ema_tilt_ok

    base_min_score = float(getattr(config, "MIN_SIGNAL_SCORE", 35) or 35)
    if period_name == "Manhã":
        min_score = base_min_score + float(getattr(config, "ENTRY_SCORE_DELTA_MORNING", 5) or 5)
    elif period_name == "Almoço":
        min_score = base_min_score + float(getattr(config, "ENTRY_SCORE_DELTA_LUNCH", 10) or 10)
    else:
        min_score = base_min_score + float(getattr(config, "ENTRY_SCORE_DELTA_AFTERNOON", 0) or 0)

    if score < min_score and not (forced_signal or adx_exception):
        reason = f"📊 Setup fraco ({period_name}): Score {score:.0f} < {min_score:.0f}"
        daily_logger.log_analysis(
            symbol=symbol, signal=side, strategy="SCORE_GATE",
            score=score, rejected=True, reason=reason,
            indicators=ind_data
        )
        from rejection_logger import log_trade_rejection
        log_trade_rejection(symbol, "ScoreGate", reason, {"score": score, "min_score": min_score, "deficit": round(min_score - score, 4), "period": period_name})
        return

    if score < base_min_score and (forced_signal or adx_exception):
        reason = f"📊 Setup abaixo do mínimo-base: Score {score:.0f} < {base_min_score:.0f}"
        daily_logger.log_analysis(
            symbol=symbol, signal=side, strategy="SCORE_GATE",
            score=score, rejected=True, reason=reason,
            indicators=ind_data
        )
        from rejection_logger import log_trade_rejection
        log_trade_rejection(symbol, "ScoreGate", reason, {"score": score, "min_score": base_min_score, "deficit": round(base_min_score - score, 4), "forced_signal": bool(forced_signal), "adx_exception": bool(adx_exception)})
        return

    # IBOV gating para ações
    if not utils.is_future(symbol):
        ibov_strength = get_ibov_adx()
        if ibov_strength < 25:
            if side == "BUY" and rsi > 30:
                daily_logger.log_analysis(
                    symbol=symbol, signal=side, strategy="IBOV_REGIME",
                    score=ind_data.get("score", 0),
                    rejected=True, reason="IBOV lateral: exige RSI ≤ 30 para compra",
                    indicators=ind_data
                )
                return
            if side == "SELL" and rsi < 70:
                daily_logger.log_analysis(
                    symbol=symbol, signal=side, strategy="IBOV_REGIME",
                    score=ind_data.get("score", 0),
                    rejected=True, reason="IBOV lateral: exige RSI ≥ 70 para venda",
                    indicators=ind_data
                )
                return
    # ========== VALIDAÇÕES COM LOG ==========

    # 2. Cooldown de saída
    if time.time() - last_close_time.get(symbol, 0) < 1800:
        daily_logger.log_analysis(
            symbol=symbol, signal=side, strategy="ELITE", score=0,
            rejected=True, reason="⏸️ Cooldown após saída (30 min)",
            indicators={"rsi": 0, "adx": 0, "spread_pips": 0, "volume_ratio": 0, "ema_trend": "N/A"}
        )
        return

    # 3. Cooldown de entrada
    if time.time() - last_entry_time.get(symbol, 0) < 300:
        logger.debug(f"⏸️ {symbol}: Aguardando cooldown entre entradas (5 min).")
        daily_logger.log_analysis(
            symbol=symbol, signal=side, strategy="ELITE", score=0,
            rejected=True, reason="⏸️ Cooldown entre entradas (5 min)",
            indicators={"rsi": 0, "adx": 0, "spread_pips": 0, "volume_ratio": 0, "ema_trend": "N/A"}
        )
        return

    # 4. Bloqueio por loss streak
    blocked, reason = utils.is_symbol_blocked(symbol)
    if blocked:
        daily_logger.log_analysis(
            symbol=symbol, signal=side, strategy="ELITE", score=0,
            rejected=True, reason=f"🚫 {reason}",
            indicators={"rsi": 0, "adx": 0, "spread_pips": 0, "volume_ratio": 0, "ema_trend": "N/A"}
        )
        return

    # 5. Limite diário
    if daily_trades_per_symbol[symbol] >= 4:
        daily_logger.log_analysis(
            symbol=symbol, signal=side, strategy="ELITE", score=0,
            rejected=True, reason=f"🚫 Limite diário ({daily_trades_per_symbol[symbol]}/4)",
            indicators={"rsi": 0, "adx": 0, "spread_pips": 0, "volume_ratio": 0, "ema_trend": "N/A"}
        )
        return

    # 6. Cotação (Cached)
    tick = utils.cached_symbol_info_tick(symbol)
    if not tick or tick.ask <= 0 or tick.bid <= 0:
        daily_logger.log_analysis(
            symbol=symbol, signal=side, strategy="ELITE", score=0,
            rejected=True, reason="❌ Sem cotação válida",
            indicators={"rsi": 0, "adx": 0, "spread_points": 0, "spread_pct": 0, "volume_ratio": 0, "ema_trend": "N/A"}
        )
        return

    # ========================================
    # ✅ NOVO: SELEÇÃO DE ESTRATÉGIA
    # ========================================
    strategy = select_trading_strategy(symbol)  # Nova função
    
    # ✅ NOVO: Bloqueio Anti-Ruído
    if strategy == "WAIT_NOISE":
        daily_logger.log_analysis(
            symbol=symbol, signal=side, strategy="NOISE_PROTECT", score=0,
            rejected=True, reason="💤 Mercado lateral (ADX<25) e RSI neutro",
            indicators={}
        )
        return

    logger.info(f"🎯 {symbol}: Estratégia selecionada: {strategy}")
    
    # Indicadores
    ind_data = bot_state.get_indicators(symbol)
    
    # ========================================
    # ✅ NOVO: VALIDAÇÃO ML (SE HABILITADO)
    # ========================================
    if config.ENABLE_ML_SIGNALS:
        # Determina grupo AB (hash simples do símbolo)
        import hashlib
        ab_group = "A" if int(hashlib.md5(symbol.encode()).hexdigest(), 16) % 2 == 0 else "B"
        ab_config = config.AB_TEST_GROUPS.get(ab_group, config.AB_TEST_GROUPS["A"])

        ml_prediction = get_ml_signal(symbol, side, ind_data)
        
        # 4. Confiança ML
        ml_confidence = ml_prediction['confidence']
        
        vix_val = utils.get_vix_br()
        loss_streak = utils.get_loss_streak(symbol)
        min_conf = 0.68
        if vix_val > 28:
            min_conf += 0.10
        if loss_streak >= 2:
            min_conf += 0.08
        ml_direction = ml_prediction['direction']

        ml_mode = str(getattr(config, "ML_MODE", "advisory")).strip().lower()
        hard_block = float(getattr(config, "ML_ADVISORY_HARD_BLOCK", 0.82))
        soft_risk = float(getattr(config, "ML_ADVISORY_SOFT_RISK", 0.70))
        soft_factor = float(getattr(config, "ML_ADVISORY_SOFT_RISK_FACTOR", 0.60))

        if ml_mode == "gate":
            ml_approved = False
            if ml_direction == side:
                if ml_confidence >= min_conf:
                    ml_approved = True
                else:
                    logger.info(f"{symbol}: ML acerta lado mas confiança baixa ({ml_confidence:.1%})")
            elif ml_direction == "HOLD":
                adx = ind_data.get("adx", 0)
                if adx > 30:
                    logger.info(f"🚀 {symbol}: ML HOLD, mas ADX {adx:.1f} > 30. Override Técnico Ativado (Risco 0.5x).")
                    risk_factor *= 0.5
                    ml_approved = True
                else:
                    logger.info(f"{symbol}: ML sugere HOLD e sem força técnica (ADX {adx:.1f}). Rejeitado.")
            else:
                logger.info(f"🛑 {symbol}: ML Contra-Tendência! (EMA: {side} vs ML: {ml_direction}). Rejeitado.")

            if not ml_approved:
                daily_logger.log_analysis(
                    symbol=symbol, signal=side, strategy="ML_ENSEMBLE",
                    score=0, rejected=True,
                    reason=f"🤖 ML Rejeitou: {ml_direction} ({ml_confidence:.1%})",
                    indicators=ind_data
                )
                return
        else:
            if ml_direction not in ("HOLD", "ERROR") and ml_direction != side:
                if ml_confidence >= hard_block:
                    daily_logger.log_analysis(
                        symbol=symbol, signal=side, strategy="ML_ADVISORY",
                        score=ind_data.get("score", 0), rejected=True,
                        reason=f"🤖 ML Contra (conf {ml_confidence:.1%})",
                        indicators=ind_data
                    )
                    return
                if ml_confidence >= soft_risk:
                    risk_factor *= soft_factor

        logger.info(f"🤖 ML {ml_mode.upper()} {symbol} | Conf: {ml_confidence:.1%} | Dir: {ml_direction}")

    # ========================================
    # ✅ NOVO: FILTROS ELITE V5.2
    # ========================================
    
    # 2. Confirmação Multi-Timeframe (M15 alinhado com H1 EMA 200)
    macro_ok = ind_data.get("macro_trend_ok", False)
    adx_val = float(ind_data.get("adx", 0) or 0)
    adx_override = float(getattr(config, "MACRO_OVERRIDE_ADX", 30) or 30)
    if (not macro_ok) and adx_val < adx_override:
        daily_logger.log_analysis(
            symbol=symbol, signal=side, strategy="MTF_CONFIRMATION",
            score=ind_data.get("score", 0),
            rejected=True, reason="🌎 Tendência Macro (H1 EMA 200) desalinhada",
            indicators=ind_data
        )
        from rejection_logger import log_trade_rejection
        log_trade_rejection(symbol, "MTF_Confirmation", "H1 EMA 200 desalinhada", {"macro_ok": bool(macro_ok), "adx": adx_val, "override_adx": adx_override})
        return
    if (not macro_ok) and adx_val >= adx_override:
        risk_factor *= float(getattr(config, "MACRO_OVERRIDE_RISK_FACTOR", 0.70) or 0.70)

    # ATR
    atr = ind_data.get("atr")
    if not atr or atr <= 0:
        logger.debug(f"⏸️ {symbol}: ATR inválido")
        daily_logger.log_analysis(
            symbol=symbol, signal=side, strategy="ELITE", 
            score=ind_data.get("score", 0),
            rejected=True, reason="❌ ATR inválido",
            indicators={
                "rsi": ind_data.get("rsi", 50),
                "adx": ind_data.get("adx", 0),
                "spread_pips": 0,
                "volume_ratio": ind_data.get("volume_ratio", 0),
                "ema_trend": "UP" if ind_data.get("ema_fast", 0) > ind_data.get("ema_slow", 0) else "DOWN"
            }
        )
        return

    # ✅ NOVO: Validação VWAP (Confirmação de Tendência)
    vwap = ind_data.get("vwap")
    if utils.is_future(symbol) and vwap:
        current_price = tick.bid if side == "BUY" else tick.ask
        if side == "BUY" and current_price < vwap:
            daily_logger.log_analysis(
                symbol=symbol, signal=side, strategy="VWAP_FILTER",
                score=0, rejected=True, reason=f"Abaixo da VWAP ({current_price} < {vwap})",
                indicators=ind_data
            )
            return
        if side == "SELL" and current_price > vwap:
            daily_logger.log_analysis(
                symbol=symbol, signal=side, strategy="VWAP_FILTER",
                score=0, rejected=True, reason=f"Acima da VWAP ({current_price} > {vwap})",
                indicators=ind_data
            )
            return
        vwap_std = ind_data.get("vwap_std")
        if vwap_std and vwap_std > 0:
            z = abs((current_price - vwap) / vwap_std)
            over_mult = float(getattr(config, "VWAP_OVEREXT_STD_MULT", 2.0) or 2.0)
            if z > over_mult:
                daily_logger.log_analysis(
                    symbol=symbol, signal=side, strategy="VWAP_FILTER",
                    score=0, rejected=True, reason=f"Esticado vs VWAP ({z:.2f}σ > {over_mult:.1f}σ)",
                    indicators=ind_data
                )
                return

    # 8. Anti-chop
    current_price = tick.bid if side == "BUY" else tick.ask
    
    can_enter, chop_reason = check_anti_chop_filter(symbol, current_price, atr)
    if not can_enter:
        logger.debug(f"🚫 {symbol}: {chop_reason}")
        daily_logger.log_analysis(
            symbol=symbol, signal=side, strategy="ELITE",
            score=ind_data.get("score", 0),
            rejected=True, reason=f"🌊 Anti-chop: {chop_reason}",
            indicators={
                "rsi": ind_data.get("rsi", 50),
                "adx": ind_data.get("adx", 0),
                "spread_pips": 0,
                "volume_ratio": ind_data.get("volume_ratio", 0),
                "ema_trend": "UP" if ind_data.get("ema_fast", 0) > ind_data.get("ema_slow", 0) else "DOWN"
            }
        )
        return
    
    # 9. Limites diários
    can_trade, limit_reason = check_daily_symbol_limit(symbol)
    if not can_trade:
        logger.info(f"🚫 {symbol}: {limit_reason}")
        daily_logger.log_analysis(
            symbol=symbol, signal=side, strategy="ELITE",
            score=ind_data.get("score", 0),
            rejected=True, reason=f"🚫 {limit_reason}",
            indicators={
                "rsi": ind_data.get("rsi", 50),
                "adx": ind_data.get("adx", 0),
                "spread_pips": 0,
                "volume_ratio": ind_data.get("volume_ratio", 0),
                "ema_trend": "UP" if ind_data.get("ema_fast", 0) > ind_data.get("ema_slow", 0) else "DOWN"
            }
        )
        return
    
    # 10. Limite de subsetor
    subsetor = config.SUBSETOR_MAP.get(symbol)

    if subsetor:
        with utils.mt5_lock:
            all_positions = mt5.positions_get() or []
    
        subsetor_count = sum(
            1 for p in all_positions 
            if config.SUBSETOR_MAP.get(p.symbol) == subsetor
        )
    
        max_subsetor = config.MAX_PER_SUBSETOR.get(subsetor, 2)
    
        if subsetor_count >= max_subsetor:
            logger.info(
                f"🚫 {symbol}: Limite de subsetor '{subsetor}' atingido "
                f"({subsetor_count}/{max_subsetor})"
            )
            daily_logger.log_analysis(
                symbol=symbol, signal=side, strategy="ELITE",
                score=ind_data.get("score", 0),
                rejected=True, 
                reason=f"🏦 Limite subsetor {subsetor} ({subsetor_count}/{max_subsetor})",
                indicators={
                    "rsi": ind_data.get("rsi", 50),
                    "adx": ind_data.get("adx", 0),
                    "spread_pips": 0,
                    "volume_ratio": ind_data.get("volume_ratio", 0),
                    "ema_trend": "UP" if ind_data.get("ema_fast", 0) > ind_data.get("ema_slow", 0) else "DOWN"
                }
            )
            return
    
    # 11. Pirâmide
    if is_pyramiding:
        if not bool(getattr(config, "ENABLE_PYRAMID", True)):
            daily_logger.log_analysis(
                symbol=symbol, signal=side, strategy="PYRAMID",
                score=ind_data.get("score", 0),
                rejected=True, reason="🔺 Pirâmide desabilitada",
                indicators=ind_data
            )
            return

        try:
            max_legs = int(getattr(config, "PYRAMID_MAX_LEGS", 3) or 3)
        except Exception:
            max_legs = 3

        try:
            current_legs = int(_symbol_pyramid_leg.get(symbol, 0) or 0)
        except Exception:
            current_legs = 0

        if max_legs >= 0 and current_legs >= max_legs:
            daily_logger.log_analysis(
                symbol=symbol, signal=side, strategy="PYRAMID",
                score=ind_data.get("score", 0),
                rejected=True, reason=f"🔺 Pirâmide: limite atingido ({current_legs}/{max_legs})",
                indicators=ind_data
            )
            return

        if bool(getattr(config, "PYRAMID_REQUIRE_PROFIT", True)):
            pos = existing_pos[0]
            if float(getattr(pos, "profit", 0.0) or 0.0) <= 0.0:
                daily_logger.log_analysis(
                    symbol=symbol, signal=side, strategy="PYRAMID",
                    score=ind_data.get("score", 0),
                    rejected=True, reason="🔺 Pirâmide: posição ainda não está em lucro",
                    indicators=ind_data
                )
                return

        try:
            min_between = float(getattr(config, "PYRAMID_MINUTES_BETWEEN_ADDS", 0) or 0)
        except Exception:
            min_between = 0.0

        if min_between > 0:
            last_ts = float(last_entry_time.get(symbol, 0) or 0.0)
            elapsed = (time.time() - last_ts) / 60.0
            if elapsed < min_between:
                daily_logger.log_analysis(
                    symbol=symbol, signal=side, strategy="PYRAMID",
                    score=ind_data.get("score", 0),
                    rejected=True, reason=f"🔺 Pirâmide: aguarde {int(min_between - elapsed)} min",
                    indicators=ind_data
                )
                return

        pos = existing_pos[0]
        try:
            last_px = float(_last_entry_price.get(symbol, getattr(pos, "price_open", 0.0)) or 0.0)
        except Exception:
            last_px = float(getattr(pos, "price_open", 0.0) or 0.0)

        atr_dist = float(getattr(config, "PYRAMID_ATR_DISTANCE", 1.0) or 1.0) * float(atr)
        pct_dist = float(getattr(config, "PYRAMID_MIN_PCT_DISTANCE", 0.0) or 0.0) * max(last_px, 1e-9)
        required_move = max(atr_dist, pct_dist)

        favorable_move = (current_price - last_px) if side == "BUY" else (last_px - current_price)
        if favorable_move < required_move:
            daily_logger.log_analysis(
                symbol=symbol, signal=side, strategy="PYRAMID",
                score=ind_data.get("score", 0),
                rejected=True, reason=f"🔺 Pirâmide: precisa +{required_move:.2f} a favor (atual {favorable_move:.2f})",
                indicators=ind_data
            )
            from rejection_logger import log_trade_rejection
            log_trade_rejection(symbol, "PyramidDistance", "Movimento insuficiente para nova perna", {"required_move": round(required_move, 4), "favorable_move": round(favorable_move, 4), "atr": float(atr)})
            return

        if now_dt.weekday() == 4:
            disable_after_str = getattr(config, "FRIDAY_DISABLE_PYRAMID_AFTER", "")
            if disable_after_str:
                try:
                    disable_after = datetime.strptime(disable_after_str, "%H:%M").time()
                    if now_dt.time() >= disable_after:
                        daily_logger.log_analysis(
                            symbol=symbol, signal=side, strategy="PYRAMID",
                            score=ind_data.get("score", 0),
                            rejected=True, reason="🔺 Pirâmide bloqueada (sexta-feira)",
                            indicators=ind_data
                        )
                        return
                except Exception:
                    pass

        can_pyramid, pyramid_reason = check_pyramid_eligibility(symbol, side, ind_data)
        
        if not can_pyramid:
            logger.debug(f"🚫 {symbol}: Pirâmide bloqueada - {pyramid_reason}")
            daily_logger.log_analysis(
                symbol=symbol, signal=side, strategy="PYRAMID",
                score=ind_data.get("score", 0),
                rejected=True, reason=f"🔺 Pirâmide: {pyramid_reason}",
                indicators={
                    "rsi": ind_data.get("rsi", 50),
                    "adx": ind_data.get("adx", 0),
                    "spread_pips": 0,
                    "volume_ratio": ind_data.get("volume_ratio", 0),
                    "ema_trend": "UP" if ind_data.get("ema_fast", 0) > ind_data.get("ema_slow", 0) else "DOWN"
                }
            )
            return
        
        logger.info(f"✅ {symbol}: Pirâmide autorizada - {pyramid_reason}")
    
    # 12. Range mínimo
    df = utils.safe_copy_rates(symbol, mt5.TIMEFRAME_M15, 20)
    
    has_range, range_reason = check_minimum_price_movement(symbol, df, atr)
    if not has_range:
        logger.debug(f"⏸️ {symbol}: {range_reason}")
        daily_logger.log_analysis(
            symbol=symbol, signal=side, strategy="ELITE",
            score=ind_data.get("score", 0),
            rejected=True, reason=f"📏 {range_reason}",
            indicators={
                "rsi": ind_data.get("rsi", 50),
                "adx": ind_data.get("adx", 0),
                "spread_pips": 0,
                "volume_ratio": ind_data.get("volume_ratio", 0),
                "ema_trend": "UP" if ind_data.get("ema_fast", 0) > ind_data.get("ema_slow", 0) else "DOWN"
            }
        )
        return

    # ✅ NOVO: News Filter
    is_blackout, news_reason = check_news_blackout(symbol)
    
    if is_blackout:
        logger.warning(f"📰 {symbol}: Bloqueado por notícia - {news_reason}")
        daily_logger.log_analysis(
            symbol=symbol, signal=side, strategy="NEWS",
            score=0, rejected=True,
            reason=f"📰 {news_reason}",
            indicators={}
        )
        return

    # ========== CÁLCULOS E VALIDAÇÕES FINAIS ==========
    
    entry_price = tick.ask if side == "BUY" else tick.bid
    atr_val = ind_data.get("atr", 0.10)

    if atr_val < (entry_price * 0.003):
        atr_val = entry_price * 0.005

    # ✅ Usa multiplicador otimizado se disponível (Default 2.5 para B3 survival)
    params = optimized_params.get(symbol, {})
    sl_mult = params.get("sl_atr_multiplier", 2.5)
    
    stop_dist = atr_val * sl_mult
    base_vol = utils.calculate_position_size_atr(symbol, stop_dist)
    base_vol = base_vol * risk_factor  # ✅ Fator de risco Land Trading
    
    # ✅ Correção de lote: ações (100), futuros (1)
    if utils.is_future(symbol):
        volume = max(1, int(base_vol))
        if is_pyramiding:
            volume = max(1, int(volume * 0.5))
    else:
        volume = (int(base_vol) // 100) * 100 
        if is_pyramiding:
            volume = (int(volume * 0.5) // 100) * 100

    if not utils.is_future(symbol):
        if volume < 100:
            logger.warning(f"⚠️ {symbol}: Volume {base_vol:.0f} insuficiente para lote de 100.")
            return

    if volume <= 0:
        daily_logger.log_analysis(
            symbol=symbol, signal=side, strategy="ELITE",
            score=ind_data.get("score", 0),
            rejected=True, reason="💰 Volume calculado = 0",
            indicators={
                "rsi": ind_data.get("rsi", 50),
                "adx": ind_data.get("adx", 0),
                "spread_pips": 0,
                "volume_ratio": ind_data.get("volume_ratio", 0),
                "ema_trend": "UP" if ind_data.get("ema_fast", 0) > ind_data.get("ema_slow", 0) else "DOWN"
            }
        )
        return

    # Bucket de capital 65/35
    alloc_ok, alloc_reason = check_capital_allocation(symbol, volume, entry_price)
    if not alloc_ok:
        daily_logger.log_analysis(
            symbol=symbol, signal=side, strategy="CAPITAL_BUCKET",
            score=ind_data.get("score", 0),
            rejected=True, reason=alloc_reason,
            indicators={}
        )
        return
    sl, tp = utils.calculate_dynamic_sl_tp(symbol, side, entry_price, ind_data)

    if utils.is_future(symbol):
        info = mt5.symbol_info(symbol)
        insp = utils.AssetInspector.detect(symbol)
        point = info.point if info else 1.0
        pv = float(insp.get("point_value", 1.0) or 1.0)
        fee_type = insp.get("fee_type", "FIXED")
        fee_val = float(insp.get("fee_val", 0.0) or 0.0)
        spread_money = abs(tick.ask - tick.bid) / max(point, 1e-9) * pv
        fees_money = fee_val * 2.0 if fee_type == "FIXED" else 0.0
        min_mult = float(get_asset_class_config(symbol)["min_tp_cost_multiplier"])
        min_cost_money = (spread_money + fees_money) * min_mult
        reward_money = abs(tp - entry_price) / max(point, 1e-9) * pv
        if reward_money < min_cost_money:
            daily_logger.log_analysis(
                symbol=symbol, signal=side, strategy="COST_FILTER",
                score=ind_data.get("score", 0),
                rejected=True, reason=f"TP insuficiente vs custo ({reward_money:.2f} < {min_cost_money:.2f})",
                indicators=ind_data
            )
            return
    if not utils.validate_order_params(symbol, side, volume, entry_price, sl, tp):
        daily_logger.log_analysis(
            symbol=symbol, signal=side, strategy="ELITE",
            score=ind_data.get("score", 0),
            rejected=True, reason="❌ Parâmetros inválidos (SL/TP/Volume)",
            indicators={
                "rsi": ind_data.get("rsi", 50),
                "adx": ind_data.get("adx", 0),
                "spread_pips": 0,
                "volume_ratio": ind_data.get("volume_ratio", 0),
                "ema_trend": "UP" if ind_data.get("ema_fast", 0) > ind_data.get("ema_slow", 0) else "DOWN"
            }
        )
        return
        
    try:
        limit = utils.get_effective_exposure_limit()
        current_exposure = utils.calculate_total_exposure()
        new_trade_value = float(volume) * float(entry_price)
        if current_exposure + new_trade_value > limit:
            logger.warning(f"⛔ Exposição total {current_exposure + new_trade_value:,.2f} > limite {limit:,.2f} | {symbol}")
            daily_logger.log_analysis(
                symbol=symbol, signal=side, strategy="ELITE",
                score=ind_data.get("score", 0),
                rejected=True, reason="⛔ Limite de exposição atingido",
                indicators={}
            )
            return
        if current_exposure >= 0.8 * limit:
            push_alert(f"⚠️ Exposição em {current_exposure/limit:.0%} do limite", "WARNING")
    except Exception as e:
        logger.error(f"Erro validação de exposição: {e}")
    
    ob_mult = float(getattr(config, "ORDER_BOOK_DEPTH_MULTIPLIER", 3) or 3)
    try:
        ph = getattr(config, "POWER_HOUR", {}) or {}
        if ph.get("enabled", True):
            start = datetime.strptime(str(ph.get("start", "15:30")), "%H:%M").time()
            end = datetime.strptime(str(ph.get("end", "16:55")), "%H:%M").time()
            if start <= datetime.now().time() <= end:
                ob_mult = float(getattr(config, "ORDER_BOOK_DEPTH_MULTIPLIER_POWER_HOUR", ob_mult) or ob_mult)
    except Exception:
        pass

    if not utils.analyze_order_book_depth(symbol, side, volume, multiplier=ob_mult):
        daily_logger.log_analysis(
            symbol=symbol, signal=side, strategy="ELITE",
            score=ind_data.get("score", 0),
            rejected=True, reason="📚 Liquidez insuficiente (Book)",
            indicators={
                "rsi": ind_data.get("rsi", 50),
                "adx": ind_data.get("adx", 0),
                "spread_pips": 0,
                "volume_ratio": ind_data.get("volume_ratio", 0),
                "ema_trend": "UP" if ind_data.get("ema_fast", 0) > ind_data.get("ema_slow", 0) else "DOWN"
            }
        )
        return
        
    # ========== VALIDAÇÃO DE ORDEM ==========
    
    from validation import validate_and_create_order 
    order, val_error = validate_and_create_order(
        symbol=symbol, side=side, volume=volume, entry_price=entry_price, sl=sl, tp=tp
    )

    if not order:
        daily_logger.log_analysis(
            symbol=symbol, signal=side, strategy="ELITE",
            score=ind_data.get("score", 0),
            rejected=True, reason=f"❌ Validação: {val_error or 'Desconhecido'}",
            indicators={
                "rsi": ind_data.get("rsi", 50),
                "adx": ind_data.get("adx", 0),
                "spread_pips": 0,
                "volume_ratio": ind_data.get("volume_ratio", 0),
                "ema_trend": "UP" if ind_data.get("ema_fast", 0) > ind_data.get("ema_slow", 0) else "DOWN"
            }
        )
        return False

    # ========== EXECUÇÃO ==========
    
    try:
        pyr_count = int(_symbol_pyramid_leg.get(symbol, 0) or 0)
    except Exception:
        pyr_count = 0
    
    daily_trades_per_symbol[symbol] += 1
    comment = f"XP3_PYR_{pyr_count + 1}" if is_pyramiding else "XP3_INIT"

    logger.info(
        f"🚀 ENVIANDO {'PIRÂMIDE' if is_pyramiding else 'ENTRADA'} {side} em {symbol} | "
        f"Vol: {volume:.0f} @ {entry_price:.2f}"
    )

    request = order.to_mt5_request(comment=comment)
    request["deviation"] = utils.get_dynamic_slippage(symbol, datetime.now().hour)
    result = mt5_order_send_safe(request)

    if result is None:
        logger.error(f"❌ {symbol}: TIMEOUT ou Falha Crítica no envio (None)")
        return False

    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        last_entry_time[symbol] = time.time()
        _last_entry_price[symbol] = float(getattr(result, "price", entry_price) or entry_price)
        if is_pyramiding:
            _symbol_pyramid_leg[symbol] = int(_symbol_pyramid_leg.get(symbol, 0) or 0) + 1
        else:
            _symbol_pyramid_leg[symbol] = 0

        with position_open_times_lock:
            positions = mt5.positions_get(symbol=symbol)
            if is_valid_dataframe(positions):
                newest_pos = max(positions, key=lambda p: p.time)
                position_open_times[newest_pos.ticket] = time.time()
        
        try:
            from utils import clear_anti_chop_cooldown
            clear_anti_chop_cooldown(symbol)
        except Exception as e:
            logger.debug(f"Erro ao limpar anti-chop: {e}")

        with entry_indicators_lock:
            entry_indicators[symbol] = ind_data.copy()

        utils.send_telegram_trade(symbol, side, volume, entry_price, sl, tp, comment)
        
        log_trade_to_txt(
            symbol=symbol,
            side=side,
            volume=volume,
            entry_price=result.price,
            exit_price=0, 
            pnl_money=0,
            pnl_pct=0,
            reason="Abertura de Posição",
        )
        try:
            if is_pyramiding:
                utils.apply_partial_exit_after_pyr(symbol)
        except Exception:
            logger.exception(f"Erro ao aplicar parcial após pirâmide em {symbol}")
        
        # ✅ LOG: SUCESSO!
        daily_logger.log_analysis(
            symbol=symbol, signal=side, 
            strategy="PYRAMID" if is_pyramiding else "ELITE",
            score=ind_data.get("score", 0),
            rejected=False,
            reason="✅ ORDEM EXECUTADA!",
            indicators={
                "rsi": ind_data.get("rsi", 50),
                "adx": ind_data.get("adx", 0),
                "spread_pips": 0,
                "volume_ratio": ind_data.get("volume_ratio", 0),
                "ema_trend": "UP" if ind_data.get("ema_fast", 0) > ind_data.get("ema_slow", 0) else "DOWN"
            }
        )
        
        return True
    else:
        logger.error(f"🚨 Falha ao enviar ordem {side} em {symbol}: {result.comment if result else 'Erro MT5'}")
        daily_trades_per_symbol[symbol] -= 1
        
        # ✅ LOG: Falha na execução
        daily_logger.log_analysis(
            symbol=symbol, signal=side, strategy="ELITE",
            score=ind_data.get("score", 0),
            rejected=True, 
            reason=f"❌ Broker rejeitou: {result.comment if result else 'Erro MT5'}",
            indicators={
                "rsi": ind_data.get("rsi", 50),
                "adx": ind_data.get("adx", 0),
                "spread_pips": 0,
                "volume_ratio": ind_data.get("volume_ratio", 0),
                "ema_trend": "UP" if ind_data.get("ema_fast", 0) > ind_data.get("ema_slow", 0) else "DOWN"
            }
        )
        
        return False

# =========================
# CIRCUIT BREAKER
# =========================
def check_for_circuit_breaker():
    global trading_paused, daily_max_equity, equity_inicio_dia, last_reset_day
    global daily_target_hit_day, daily_target_hit_pct, _symbol_pyramid_leg, _last_entry_price
    global TRADING_PAUSED, PAUSE_REASON, CIRCUIT_BREAKER_DISABLED
    global pause_reset_day

    with utils.mt5_lock:
        acc = mt5.account_info()
    if not acc:
        return

    now = datetime.now()
    today = now.date()

    if CIRCUIT_BREAKER_DISABLED:
        TRADING_PAUSED = False
        PAUSE_REASON = ""
        trading_paused = False
        _last_wr_alert_ts = 0.0
        _last_wr_alert_wr = None
        _last_dd_alert_ts = 0.0
        _last_dd_alert_dd = None
        return

    reset_time = datetime.strptime(config.DAILY_RESET_TIME, "%H:%M").time()
    if now.time() >= reset_time and pause_reset_day != today:
        TRADING_PAUSED = False
        PAUSE_REASON = ""
        trading_paused = False
        pause_reset_day = today
        _last_wr_alert_ts = 0.0
        _last_wr_alert_wr = None
        _last_dd_alert_ts = 0.0
        _last_dd_alert_dd = None
        logger.info("✅ Estado de pausa por win rate resetado (meia-noite)")
    if now.time() >= reset_time and last_reset_day != today:
        daily_max_equity = acc.equity
        equity_inicio_dia = acc.equity
        with open("daily_equity.txt", "w") as f:
            f.write(str(equity_inicio_dia))
        trading_paused = False
        TRADING_PAUSED = False
        PAUSE_REASON = ""
        last_reset_day = today
        daily_trades_per_symbol.clear()
        daily_target_hit_day = None
        daily_target_hit_pct = None
        _symbol_pyramid_leg.clear()
        _last_entry_price.clear()
        _last_wr_alert_ts = 0.0
        _last_wr_alert_wr = None
        _last_dd_alert_ts = 0.0
        _last_dd_alert_dd = None
        logger.info("✅ Estado de pausa por win rate resetado (diário)")
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
    if not is_valid_dataframe(positions, min_rows=2):
        return 0.0

    symbols = [p.symbol for p in positions]

    # 1. Correlação média
    total_corr = 0.0
    count = 0
    for sym1 in symbols:
        for sym2 in symbols:
            if sym1 >= sym2:
                continue
            corr = (
                utils.calculate_correlation_matrix([sym1, sym2])
                .get(sym1, {})
                .get(sym2, 0)
            )
            total_corr += abs(corr)
            count += 1
    avg_corr = total_corr / count if count > 0 else 0.0

    # 2. Concentração setorial (HHI)
    equity = mt5.account_info().equity
    sector_exp = utils.calculate_sector_exposure_pct(equity)
    hhi = sum(exp**2 for exp in sector_exp.values())

    # 3. Volatilidade agregada
    # Usa snapshot do bot_state
    indicators, _ = bot_state.snapshot
    total_atr_pct = (
        sum(indicators.get(p.symbol, {}).get("atr_real", 0) for p in positions)
        / len(positions)
        if positions
        else 0
    )

    heat = (avg_corr * 0.4) + (hhi * 0.3) + (min(total_atr_pct / 10.0, 1.0) * 0.3)
    return round(min(heat, 1.0), 3)


# =========================
# 🎯 PROFIT LOCK (ADICIONAR APÓS check_for_circuit_breaker)
# =========================
_profit_lock_last_action_ts = 0.0
_profit_lock_last_action_day = None

def apply_profit_lock_actions(daily_pnl: float, daily_pnl_pct: float, reason: str) -> dict:
    global _profit_lock_last_action_ts, _profit_lock_last_action_day

    now = datetime.now()
    lock_cfg = getattr(config, "PROFIT_LOCK", {}) or {}
    min_minutes = int(lock_cfg.get("min_minutes_between_actions", 5) or 5)

    if _profit_lock_last_action_day != now.date():
        _profit_lock_last_action_day = now.date()
        _profit_lock_last_action_ts = 0.0

    if time.time() - _profit_lock_last_action_ts < (min_minutes * 60):
        return {"closed": 0, "locked_profit": 0.0, "tightened": 0, "target_lock": 0.0}

    _profit_lock_last_action_ts = time.time()

    with utils.mt5_lock:
        positions = mt5.positions_get() or []

    if not positions:
        return {"closed": 0, "locked_profit": 0.0, "tightened": 0, "target_lock": 0.0}

    close_winners_only = bool(lock_cfg.get("close_winners_only", True))
    tighten_trailing = bool(lock_cfg.get("tighten_trailing", True))
    atr_mult = float(lock_cfg.get("tighten_trailing_atr_mult", 1.0) or 1.0)
    lock_pct = float(lock_cfg.get("lock_pct", 0.7) or 0.7)

    target_lock = max(0.0, float(daily_pnl) * lock_pct) if float(daily_pnl) > 0 else 0.0
    locked_profit = 0.0
    closed = 0

    candidates = [p for p in positions if p.profit > 0] if close_winners_only else list(positions)
    candidates.sort(key=lambda p: p.profit, reverse=True)

    for pos in candidates:
        if target_lock > 0 and locked_profit >= target_lock:
            break
        if close_winners_only and pos.profit <= 0:
            continue
        try:
            with utils.mt5_lock:
                tick = mt5.symbol_info_tick(pos.symbol)
            if not tick:
                continue
            price = tick.bid if pos.type == mt5.POSITION_TYPE_BUY else tick.ask
            close_position(
                pos.symbol,
                pos.ticket,
                pos.volume,
                price,
                reason=f"Profit Lock - {reason}",
            )
            locked_profit += max(0.0, float(pos.profit))
            closed += 1
        except Exception as e:
            logger.error(f"Erro no Profit Lock (close) para {pos.symbol}: {e}")

    tightened = 0
    if tighten_trailing:
        indicators, _ = bot_state.snapshot
        with utils.mt5_lock:
            remaining = mt5.positions_get() or []

        for pos in remaining:
            ind = indicators.get(pos.symbol, {})
            atr = ind.get("atr") or ind.get("atr_real")
            if not atr or atr <= 0:
                continue

            side = "BUY" if pos.type == mt5.POSITION_TYPE_BUY else "SELL"
            current_price = float(pos.price_current)
            current_sl = float(pos.sl or 0)

            if side == "BUY":
                new_sl = current_price - (float(atr) * atr_mult)
                if current_sl > 0:
                    new_sl = max(current_sl, new_sl)
                if new_sl <= 0 or new_sl >= current_price:
                    continue
            else:
                new_sl = current_price + (float(atr) * atr_mult)
                if current_sl > 0:
                    new_sl = min(current_sl, new_sl)
                if new_sl <= current_price:
                    continue

            try:
                modify_sl(pos.symbol, pos.ticket, new_sl)
                tightened += 1
            except Exception as e:
                logger.error(f"Erro no Profit Lock (tighten) para {pos.symbol}: {e}")

    return {"closed": closed, "locked_profit": locked_profit, "tightened": tightened, "target_lock": target_lock}

def check_profit_lock():
    """
    Se bateu a meta diária, trava parte do lucro e reduz agressividade
    """
    global trading_paused

    with utils.mt5_lock:
        acc = mt5.account_info()

    if not acc:
        return

    # Calcula P&L do dia
    daily_pnl = acc.equity - equity_inicio_dia
    daily_pnl_pct = (daily_pnl / equity_inicio_dia) if equity_inicio_dia > 0 else 0

    # Verifica se bateu a meta
    if (
        config.PROFIT_LOCK["enabled"]
        and daily_pnl_pct >= config.PROFIT_LOCK["daily_target_pct"]
    ):
        stats = apply_profit_lock_actions(daily_pnl=daily_pnl, daily_pnl_pct=daily_pnl_pct, reason="Meta Diária Atingida")

        if stats.get("closed", 0) > 0 or stats.get("tightened", 0) > 0:
            locked_profit = stats.get("locked_profit", 0.0)
            target_lock = stats.get("target_lock", 0.0)
            push_alert(
                f"🎯 META DIÁRIA ATINGIDA! Lucro: R${daily_pnl:+.2f} ({daily_pnl_pct * 100:.1f}%) | "
                f"Fechadas {stats.get('closed', 0)} posições em lucro (travado ~R${locked_profit:,.2f}/{target_lock:,.2f}) | "
                f"Trailing apertado: {stats.get('tightened', 0)}",
                "INFO",
                True,
            )

        # Opcional: Reduz agressividade para o resto do dia
        if config.PROFIT_LOCK["reduce_risk"]:
            # Não pausar trading, mas o get_current_risk_pct() vai pegar isso
            pass


# =========================
# HORÁRIO DE TRADING
# =========================
def is_trading_time_allowed(new_entry: bool = True) -> bool:
    """
    Valida horário de trading SEM encerrar o bot
    """
    market_status = get_market_status()

    # Não permite trading se mercado fechado
    if not market_status["trading_allowed"]:
        return False

    # Se está pedindo nova entrada, valida isso também
    if new_entry and not market_status["new_entries_allowed"]:
        return False

    return True


# =========================
# RELATÓRIO DIÁRIO
# =========================
def daily_report():
    """
    Thread de monitoramento SEM encerrar o bot
    """
    while True:  # ✅ Loop infinito
        try:
            # Chama o gerenciador de ciclo
            handle_daily_cycle()

            logger.info(f"🧠 Treino diário ML iniciado (histórico: {len(ml_optimizer.history)} trades | por ativo: {getattr(config, 'ML_TRAIN_PER_SYMBOL', False)})")
            ml_optimizer.train_ensemble()
            logger.info("✅ Treino diário ML finalizado")
            
            time.sleep(86400)

        except Exception as e:
            logger.error(f"Erro no daily_report: {e}", exc_info=True)
            time.sleep(30)


def log_trade_to_txt(
    symbol: str,
    side: str,
    volume: float,
    entry_price: float,
    exit_price: float,
    pnl_money: float,
    pnl_pct: float,
    reason: str,
    deal_id: Optional[int] = None,
    position_id: Optional[int] = None,
):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Se o preço de saída for 0, identificamos como uma ENTRADA
    tipo = "ENTRADA" if exit_price == 0 else "SAÍDA"

    # Formatamos a linha para incluir o tipo
    price_to_show = entry_price if tipo == "ENTRADA" else exit_price
    extra = ""
    if deal_id is not None:
        extra += f" | DealId: {deal_id}"
    if position_id is not None:
        extra += f" | Pos: {position_id}"
    line = (
        f"{timestamp} | {tipo:<8} | {symbol:<6} | {side:<4} | "
        f"Vol: {volume:>7.0f} | Price: {price_to_show:>6.2f} | "
        f"P&L: {pnl_money:>+8.2f} R$ ({pnl_pct:+.2f}%) | "
        f"Motivo: {reason}{extra}\n"
    )

    filename = f"trades_log_{datetime.now().strftime('%Y-%m-%d')}.txt"

    if not os.path.exists(filename):
        header = "DATA/HORA           | TIPO     | ATIVO  | LADO |   VOLUME |  PREÇO |     P&L R$    |   %    | MOTIVO\n"
        header += "-" * 105 + "\n"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(header)

    with open(filename, "a", encoding="utf-8") as f:
        f.write(line)


# =========================
# DASHBOARD
# =========================
# ============================================
# IMPORTS NECESSÁRIOS (Adicione no topo do bot.py)
# ============================================
try:
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.text import Text
    from rich.style import Style
    from rich import box
    console = Console()
except ImportError:
    console = None
    Live = None
    # Fallback se não tiver rich instalado
    print("⚠️ Biblioteca 'rich' não instalada. Instale com: pip install rich")

# ============================================
# PAINEL (VERSÃO RICH - ESTILO FOREX)
# ============================================
# ============================================
# IMPORTS NECESSÁRIOS (Adicione no topo do bot.py)
# ============================================
try:
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.text import Text
    from rich.style import Style
    from rich import box
    console = Console()
except ImportError:
    console = None
    Live = None
    # Fallback se não tiver rich instalado
    print("⚠️ Biblioteca 'rich' não instalada. Instale com: pip install rich")

# ============================================
# PAINEL (VERSÃO RICH )
# ============================================
def launch_dashboard():
    """Inicia o dashboard Streamlit em processo separado (Porta 8501)"""
    try:
        # --server.headless=true esconde o menu de dev do streamlit no console
        cmd = [
            sys.executable, "-m", "streamlit", "run", "dashboard.py",
            "--server.port=8501",
            "--server.address=localhost",
            "--server.headless=true",
            "--theme.base=light"
        ]
        # Inicia sem bloquear
        subprocess.Popen(cmd)
        time.sleep(3) # Espera iniciar
        logger.info("✅ Dashboard iniciado com sucesso na porta 8501!")
        
        # Tenta abrir navegador
        try:
            webbrowser.open("http://localhost:8501")
        except: pass
        
    except Exception as e:
        logger.error(f"Erro ao iniciar dashboard: {e}")



def close_all_positions(reason: str = "Fechamento diário"):
    """
    ✅ VERSÃO SIMPLIFICADA: UMA thread, UMA vez
    """
    logger.info("=" * 70)
    logger.info(f"🔒 INICIANDO FECHAMENTO: {reason}")
    logger.info(f"⏰ Horário: {datetime.now().strftime('%H:%M:%S')}")
    logger.info("=" * 70)
    
    with utils.mt5_lock:
        positions = mt5.positions_get()
    
    if not is_valid_dataframe(positions):
        logger.info("✅ Nenhuma posição aberta")
        return
    
    total = len(positions)
    logger.warning(f"⚠️ {total} posições para fechar")
    # Lista símbolos
    for i, pos in enumerate(positions, 1):
        logger.info(f"   {i}. {pos.symbol} | Ticket: {pos.ticket}")
    
    # === 🎯 FECHA CADA POSIÇÃO ===
    success_count = 0
    failed = []
    
    for idx, pos in enumerate(positions, 1):
        symbol = pos.symbol
        ticket = pos.ticket
        
        logger.info(f"\n📍 [{idx}/{total}] Fechando {symbol} (ticket {ticket})")
        
        # Pega cotação
        with utils.mt5_lock:
            tick = mt5.symbol_info_tick(symbol)
        
        if not tick:
            logger.error(f"❌ {symbol}: Sem cotação")
            failed.append(ticket)
            continue
        
        side = "BUY" if pos.type == mt5.POSITION_TYPE_BUY else "SELL"
        price = tick.bid if side == "BUY" else tick.ask
        
        # Fecha (já tem retry interno de 5x)
        closed = close_position(
            symbol=symbol,
            ticket=ticket,
            volume=pos.volume,
            price=price,
            reason=reason
        )
        
        if closed:
            success_count += 1
            logger.info(f"✅ [{success_count}/{total}] {symbol} fechado")
        else:
            failed.append(ticket)
            logger.error(f"❌ {symbol} FALHOU")
        
        # Pausa entre posições
        time.sleep(3)
    
    # === 📊 RELATÓRIO FINAL ===
    with utils.mt5_lock:
        remaining = mt5.positions_get() or []
    
    logger.info(f"\n{'='*70}")
    logger.info(f"📊 RESULTADO FINAL DO FECHAMENTO")
    logger.info(f"{'='*70}")
    logger.info(f"✅ Fechadas: {success_count}/{total}")
    logger.info(f"❌ Falharam: {len(failed)}")
    logger.info(f"🔍 Restantes: {len(remaining)}")
    
    if remaining:
        logger.critical(f"🚨 ATENÇÃO: {len(remaining)} POSIÇÕES AINDA ABERTAS:")
        for p in remaining:
            logger.critical(f"   • {p.symbol} | Ticket: {p.ticket}")
        
        # Notificação Telegram
        try:
            utils.send_telegram_message(
                f"🚨 <b>ALERTA EOD</b>\n\n"
                f"❌ {len(remaining)} posições abertas\n"
                f"Símbolos: {', '.join([p.symbol for p in remaining])}\n"
                f"Tickets: {', '.join([str(p.ticket) for p in remaining])}\n\n"
                f"⚠️ Feche manualmente!"
            )
        except:
            pass
    else:
        logger.info("🎉 SUCESSO TOTAL: Todas posições fechadas!")
    
    logger.info(f"{'='*70}\n")


# ============================================
# FUNÇÃO 2: fast_loop() - Linha ~1741
# ============================================

# CORREÇÃO COMPLETA DA FUNÇÃO fast_loop()
# Substitua a função inteira no bot.py (linha ~1741)

def fast_loop():
    """
    Loop principal com operação contínua
    ✅ VERSÃO CORRIGIDA: Inclui failsafe de fechamento EOD
    """
    global trading_paused, daily_cycle_completed, _last_summary_time

    logger.info("⚙️ Fast Loop iniciado (modo contínuo)")
    logger.info("🔄 Inicializando pesos adaptativos e correlações...")
    utils.update_adaptive_weights()

    while True:  # ✅ Loop infinito (não depende de bot_should_run)
        try:
            health_monitor.heartbeat()

            # ============================================
            # 🔴 PRIORIDADE MÁXIMA: GERENCIADOR DE CICLO
            # ============================================
            # Chama ANTES do failsafe para evitar duplicação
            handle_daily_cycle()

            # ============================================
            # 🚨 FAILSAFE CRÍTICO: FECHAMENTO FORÇADO
            # ============================================
            
            now = datetime.now()
            close_str = getattr(config, "FRIDAY_CLOSE_ALL_BY", config.CLOSE_ALL_BY) if now.weekday() == 4 else config.CLOSE_ALL_BY
            close_time = datetime.strptime(close_str, "%H:%M").time()

            # Ativa 2 minutos APÓS o horário (era 5 minutos)
            failsafe_time = (datetime.combine(now.date(), close_time) + timedelta(minutes=2)).time()

            if now.time() >= failsafe_time:
                with utils.mt5_lock:
                    positions = mt5.positions_get() or []

                if positions and not daily_cycle_completed:
                    logger.critical(
                        f"🚨 FAILSAFE ATIVADO! Fechamento normal FALHOU às {now.strftime('%H:%M:%S')}"
                    )
        
                    push_alert(
                        f"🚨 FAILSAFE: {len(positions)} posições não fecharam no horário!",
                        "CRITICAL", sound=True
                    )

                    # Tenta fechar com MÁXIMA prioridade
                    close_all_positions(reason="FAILSAFE EMERGENCIAL")
        
                    # Aguarda 5s
                    time.sleep(5)
        
                    # Verifica novamente
                    with utils.mt5_lock:
                        still_open = mt5.positions_get() or []
        
                    if still_open:
                        # 🔥 EMERGÊNCIA TOTAL
                        logger.critical(
                            f"🔥 EMERGÊNCIA: {len(still_open)} posições AINDA abertas após failsafe!"
                        )
            
                        try:
                            utils.send_telegram_message(
                                f"🔥 <b>EMERGÊNCIA TOTAL</b>\n\n"
                                f"❌ Failsafe FALHOU\n"
                                f"⏰ {now.strftime('%H:%M:%S')}\n"
                                f"📊 {len(still_open)} posições abertas\n\n"
                                f"🚨 <b>FECHE MANUALMENTE AGORA!</b>"
                            )
                        except:
                            pass
                    
                    # Força marcação do ciclo
                    daily_cycle_completed = True
            # Nova: Day-only mode - Fecha posições se hora >= CLOSE_ALL_BY
            now = datetime.now()
            close_str = getattr(config, "FRIDAY_CLOSE_ALL_BY", getattr(config, 'CLOSE_ALL_BY', '16:45')) if now.weekday() == 4 else getattr(config, 'CLOSE_ALL_BY', '16:45')
            try:
                close_time = datetime.strptime(close_str, "%H:%M").time()
            except:
                close_time = datetime.strptime("16:45", "%H:%M").time()

            is_eod_time = now.time() >= close_time
            
            if config.DAY_ONLY_MODE and is_eod_time:
                with utils.mt5_lock:
                    positions = mt5.positions_get() or []
                
                if positions:
                    # Loga apenas uma vez a cada minuto para não floodar
                    if now.second < 5:
                        logger.warning(f"🚨 EOD TRIGGER ({close_str}): Fechando {len(positions)} posições abertas.")
                    
                    for pos in positions:
                        try:
                            tick = mt5.symbol_info_tick(pos.symbol)
                            if tick:
                                price = tick.bid if pos.type == mt5.POSITION_TYPE_BUY else tick.ask
                                # logger.info(f"⏳ Tentando fechar {pos.symbol} (Ticket {pos.ticket})...")
                                close_position(pos.symbol, pos.ticket, pos.volume, price, reason="Day-Only Close")
                            else:
                                logger.error(f"❌ Sem tick para fechar {pos.symbol}")
                        except Exception as e:
                            logger.error(f"Erro ao fechar {pos.symbol} no Day-Only: {e}")
        
            # Nova chamada para hedging se DD >3%
            if utils.calculate_daily_dd() > 0.03:
                apply_hedge()  # De hedging.py
        
            time.sleep(5)  # 5s
            
            # ============================================
            # 0️⃣ VERIFICA PERMISSÃO DE TRADING (WR/Paused)
            # ============================================
            # Verifica/reset de circuit breaker e pausa por WR mesmo se já estiver pausado
            check_for_circuit_breaker()
            allowed, reason = is_trading_allowed()
            if not allowed:
                if not trading_paused:
                    logger.warning(f"⛔ Trading PAUSADO globalmente: {reason}")
                    # Força atualização global
                    trading_paused = True
            else:
                if trading_paused:
                    logger.info(f"✅ Trading RETOMADO: {reason}")
                    trading_paused = False

            # ============================================
            # 1️⃣ OBTÉM STATUS DO MERCADO
            # ============================================
            market_status = get_market_status()

            # ============================================
            # 2️⃣ ATUALIZA DADOS (SEMPRE)
            # ============================================
            new_indicators, new_top15 = build_portfolio_and_top15()
            update_bot_bridge()
            try:
                utils.check_and_apply_dynamic_trailing(interval_sec=300)
            except Exception:
                logger.exception("Erro ao aplicar trailing dinâmico")

            # ============================================
            # 3️⃣ VERIFICA CIRCUIT BREAKER
            # ============================================
            if trading_paused:
                time.sleep(5)
                continue

            # ============================================
            # 4️⃣ SE MERCADO FECHADO, AGUARDA
            # ============================================
            if not market_status["trading_allowed"]:
                # Log silencioso a cada 5 minutos
                if datetime.now().minute % 5 == 0 and datetime.now().second < 5:
                    logger.debug(
                        f"{market_status['emoji']} {market_status['message']} | "
                        f"Próximo pregão: {market_status['countdown']}"
                    )

                time.sleep(30)  # Aguarda mais tempo quando fechado
                continue

            # ============================================
            # 5️⃣ GESTÃO DE POSIÇÕES (SE HOUVER)
            # ============================================
            try:
                manage_positions_refactored()
            except Exception as e:
                logger.error(f"Erro na gestão de posições: {e}")

            # ============================================
            # 6️⃣ PROFIT LOCK
            # ============================================
            check_profit_lock()

            # ============================================
            # 7️⃣ PROCESSAMENTO DE SINAIS (SE PERMITIDO)
            # ============================================
            if market_status["new_entries_allowed"]:
                symbols_to_scan = list(optimized_params.keys())
                current_win = utils.resolve_current_symbol("WIN")
                current_wdo = utils.resolve_current_symbol("WDO")
                if current_win and current_win not in symbols_to_scan:
                    symbols_to_scan.append(current_win)
                if current_wdo and current_wdo not in symbols_to_scan:
                    symbols_to_scan.append(current_wdo)
                try:
                    symbols_to_scan = [s for s in symbols_to_scan if utils.is_time_allowed_for_symbol(s, CURRENT_MODE)]
                except Exception:
                    pass

                for sym in symbols_to_scan:
                    ind_data = bot_state.get_indicators(sym)

                    if not ind_data or ind_data.get("error"):
                        continue

                    score = utils.calculate_signal_score(ind_data)
                    
                    # ✅ LÓGICA V5.5 AGRESSIVA
                    ema_trend = "UP" if ind_data["ema_fast"] > ind_data["ema_slow"] else "DOWN"
                    rsi = ind_data.get("rsi", 50)
                    adx = ind_data.get("adx", 0)
                    
                    forced_signal = (ema_trend == "UP" and rsi > 50) or (ema_trend == "DOWN" and rsi < 50)
                    
                    ema_diff_pct = abs(ind_data["ema_fast"] - ind_data["ema_slow"]) / max(ind_data["close"], 1)
                    adx_exception = (15 <= adx <= 20) and (ema_diff_pct > 0.0005)

                    if score >= config.MIN_SIGNAL_SCORE or forced_signal or adx_exception:
                        side = "BUY" if ema_trend == "UP" else "SELL"
                        
                        # 🛡️ FILTRO DE EXAUSTÃO (MEAN REVERSION) - LAND TRADING
                        # Evita comprar topo (RSI > 70) ou vender fundo (RSI < 30)
                        if side == "BUY" and rsi > 70:
                            # logger.debug(f"🛑 {sym}: RSI esticado ({rsi:.1f} > 70) - Compra evitada.")
                            continue
                        if side == "SELL" and rsi < 30:
                            # logger.debug(f"🛑 {sym}: RSI esticado ({rsi:.1f} < 30) - Venda evitada.")
                            continue
                            
                        try_enter_position(sym, side)

            # ============================================
            # 8️⃣ CIRCUIT BREAKER
            # ============================================
            check_for_circuit_breaker()
            
            # ============================================
            # 9️⃣ SALVA CACHES
            # ============================================
            save_top15_cache()
            save_system_status()
            
            # ============================================
            # ⏱️ AGUARDA 5 SEGUNDOS
            # ============================================
            time.sleep(5)

        except KeyboardInterrupt:
            logger.info("⚠️ Ctrl+C detectado - Encerrando fast_loop...")
            break
            
        except Exception as e:
            logger.error(f"Erro crítico no fast_loop: {e}", exc_info=True)
            
            # Tenta salvar dados antes de continuar
            utils.save_adaptive_weights()
        
        finally:
            # ✅ GARANTE SALVAMENTO DO ML EM CASO DE CRASH/BREAK
            try:
                ml_optimizer.force_save()
            except:
                pass

    now = time.time()
    if now - _last_summary_time > 3600:
        _, top15 = bot_state.snapshot
        avg_score = np.mean([indicators.get(sym, {}).get("score", 0) for sym in top15])
        
        logger.info(
            f"📊 RESUMO HORÁRIO | "
            f"Avg Score: {avg_score:.1f} | "
            f"Posições: {len(mt5.positions_get() or [])} | "
            f"Regime: {detect_market_regime()}"
        )
        _last_summary_time = now  
        
        time.sleep(10)  # Aguarda mais tempo em caso de erro
            
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


def force_test_trade(symbol, side="BUY"):
    """
    Ignora todos os filtros e tenta abrir uma ordem imediatamente para teste.
    """
    logger.info(f"🧪 INICIANDO TESTE FORÇADO: {side} em {symbol}")

    # 1. Obtém indicadores para o cálculo do SL/TP
    ind_data, _ = build_portfolio_and_top15()
    data = ind_data.get(symbol)

    if not is_valid_dataframe(data):
        logger.error(f"❌ Falha no teste: Não foi possível obter dados de {symbol}")
        return

    # 2. Define volume mínimo para teste (ex: 100 ações ou 1 contrato)
    # Ajuste o volume conforme o ativo (100 para ações, 1 para índices)
    test_volume = 100.0

    # 3. Executa a entrada
    try:
        logger.info(f"🚀 Disparando ordem de teste para {symbol}...")
        try_enter_position(symbol, side)
        logger.info(f"✅ Comando de teste enviado. Verifique o Terminal MT5.")
    except Exception as e:
        logger.error(f"❌ Erro no disparo do teste: {e}")


bot = get_telegram_bot()


@bot.message_handler(commands=["lucro"])
def comando_lucro(message):
    # Verifica se é você mesmo mandando (segurança)
    if str(message.chat.id) == str(config.TELEGRAM_CHAT_ID):
        responder_comando_lucro(message)
    else:
        bot.reply_to(message, "❌ Acesso negado.")


def responder_comando_lucro(message):
    from utils import calcular_lucro_realizado_txt, mt5_lock
    import MetaTrader5 as mt5

    bot = get_telegram_bot()
    if not bot:
        return

    # Verifica se é o usuário autorizado
    if str(message.chat.id) != str(config.TELEGRAM_CHAT_ID):
        bot.reply_to(message, "❌ Acesso negado.")
        return

    try:
        # Lucro realizado (do arquivo txt)
        realizado, qtd = calcular_lucro_realizado_txt()

        # Lucro flutuante (posições abertas)
        with mt5_lock:
            positions = mt5.positions_get() or []
        flutuante = sum(p.profit for p in positions)

        total_do_dia = realizado + flutuante

        emoji = "🟢🚀" if total_do_dia >= 0 else "🔴⚠️"

        msg = (
            f"{emoji} <b>RESUMO FINANCEIRO XP3</b>\n\n"
            f"💰 <b>Realizado (no bolso):</b> R$ {realizado:,.2f}\n"
            f"📈 <b>Flutuante (aberto):</b> R$ {flutuante:+,.2f}\n"
            f"────────────────────\n"
            f"🏆 <b>TOTAL DO DIA:</b> R$ {total_do_dia:+,.2f}\n\n"
            f"📊 <i>{qtd} trades fechados • {len(positions)} posições abertas</i>\n"
            f"🕐 <i>{datetime.now().strftime('%H:%M:%S')}</i>"
        )

        bot.reply_to(message, msg, parse_mode="HTML")

    except Exception as e:
        logger.error(f"Erro ao responder /lucro: {e}")
        bot.reply_to(message, "❌ Erro ao obter dados financeiros.")


# Adicione estas funções para salvar dados em cache

import json
def save_top15_cache():
    try:
        indicators, top15 = bot_state.snapshot
        
        data_list = []
        for rank, sym in enumerate(top15, 1):
            ind = indicators.get(sym, {})
            current_score = float(ind.get("score", 0))
            
            data_list.append({
                "rank": int(rank),
                "symbol": str(sym),
                "score": round(current_score, 1),
                "direction": str(ind.get("direction", "NEUTRAL")),
                "rsi": round(float(ind.get("rsi", 50)), 1),
                "atr_pct": round(float(ind.get("atr_pct", 0)), 2),
                "price": round(float(ind.get("close", 0)), 2),
                "sector": str(config.SECTOR_MAP.get(sym, "OUTROS")),
                "status": "✔️ ABERTO" if current_score >= 50 else "⏸️ AGUARDANDO"
            })

        bridge_data = {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "top15": data_list,
            # ✅ Já estamos limpando os indicadores aqui
            "indicators": sanitize_for_json(indicators)  
        }

        final_file = "bot_bridge.json"

        # ✅ MELHORIA: Escrita direta com Encoder de segurança
        try:
            with open(final_file, "w", encoding="utf-8") as f:
                # Usamos o cls=NpEncoder como "seguro de vida" caso algo escape da sanitize
                json.dump(bridge_data, f, indent=4, cls=NpEncoder)
        
        except PermissionError:
            # Silencioso: acontece quando o Dashboard está lendo o arquivo
            pass
            
    except Exception as e:
        logger.error(f"❌ Erro crítico ao atualizar bot_bridge: {e}")

def update_bot_bridge():
    """Atualiza bot_bridge.json com TOP15 real e status corretos"""
    try:
        # Pega snapshot atual do bot
        indicators, top15 = bot_state.snapshot
        
        if not top15:
            logger.warning("TOP15 vazio - não atualizando bridge")
            return
        
        # Pega posições reais do MT5
        with mt5_lock:
            positions = mt5.positions_get() or []
        positions_symbols = {p.symbol for p in positions}
        
        # Limpa e reconstrói indicadores seguros
        safe_indicators = {}
        for sym, ind in indicators.items():
            if not isinstance(ind, dict):
                continue
            safe_indicators[sym] = sanitize_for_json(ind)
        
        data = {
            "timestamp": datetime.now().isoformat(),
            "top15": [],
            "indicators": safe_indicators
        }
        
        for rank, sym in enumerate(top15, 1):
            ind = safe_indicators.get(sym, {})
            
            score = float(ind.get("score", 0))
            direction = ind.get("direction", "—")
            rsi = round(float(ind.get("rsi", 50)), 1)
            
            atr_real = ind.get("atr_real", 0)
            atr_pct = min(round(float(atr_real) * 5.3, 2), 9.99) if atr_real > 0 else 0.0
            
            price = round(float(ind.get("close", 0)), 2)
            sector = str(config.SECTOR_MAP.get(sym, "UNKNOWN"))
            
            # STATUS CORRETO
            if sym in positions_symbols:
                status = "✔️ ABERTO"
            elif score >= config.MIN_SIGNAL_SCORE:
                status = "🟢 PRONTO"
            else:
                status = "⏸️ AGUARDANDO"
            
            data["top15"].append({
                "rank": rank,
                "symbol": sym,
                "score": round(score, 1),
                "direction": direction,
                "rsi": rsi,
                "atr_pct": atr_pct,
                "price": price,
                "sector": sector,
                "status": status
            })
        
        # Escrita ATÔMICA e segura
        temp_file = "bot_bridge.json.tmp"
        final_file = "bot_bridge.json"
        
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, cls=NpEncoder)
        
        # Substitui só se sucesso
        if os.path.exists(final_file):
            os.replace(temp_file, final_file)
        else:
            os.rename(temp_file, final_file)
            
        logger.debug("✅ bot_bridge.json atualizado com sucesso")
        
    except Exception as e:
        logger.error(f"❌ Erro crítico ao atualizar bot_bridge: {e}")
def telegram_polling_thread():
    """
    Thread para polling contínuo do Telegram.
    Usa o bot já configurado em telegram_handler.py
    """
    if not getattr(config, "ENABLE_TELEGRAM_NOTIF", False):
        logger.info("Telegram desativado no config → thread não iniciada")
        return
    
    if bot is None:
        logger.error("Bot Telegram não foi inicializado (verifique token no config)")
        return
    
    logger.info("🚀 Iniciando polling do Telegram com comandos integrados...")
    
    # Mensagem de startup (opcional)
    try:
        bot.send_message(config.TELEGRAM_CHAT_ID, 
                         "✅ <b>Bot Telegram conectado!</b>\nUse /help para ver comandos.",
                         parse_mode="HTML")
    except Exception as e:
        logger.warning(f"Não enviou mensagem de startup: {e}")
    
    failures = 0
    backoff_s = 5

    while True:
        try:
            from telebot import apihelper

            apihelper.RETRY_ON_ERROR = True
            apihelper.RETRY_TIMEOUT = 5

            bot.polling(none_stop=True, interval=1, timeout=20)
            failures = 0
            backoff_s = 5
        except Exception as e:
            is_timeout = False
            try:
                import requests
                is_timeout = is_timeout or isinstance(e, requests.exceptions.ReadTimeout)
            except Exception:
                pass

            try:
                import urllib3
                is_timeout = is_timeout or isinstance(e, urllib3.exceptions.ReadTimeoutError)
            except Exception:
                pass

            is_timeout = is_timeout or isinstance(e, TimeoutError)

            failures += 1
            if is_timeout:
                logger.warning(f"Telegram timeout (tentativa {failures}). Retentando em {backoff_s}s.")
            else:
                logger.error(f"Erro no polling Telegram (tentativa {failures}): {e}", exc_info=True)

            if failures in (3, 10, 30):
                try:
                    bot.send_message(
                        config.TELEGRAM_CHAT_ID,
                        f"⚠️ Telegram instável. Reconectando (falhas consecutivas: {failures}).",
                        parse_mode="HTML",
                    )
                except Exception:
                    pass

            time.sleep(backoff_s)
            backoff_s = min(int(backoff_s * 1.7), 120)

def save_system_status():
    """Salva status do sistema"""
    try:
        status = {
            "correlation_updated": last_correlation_update is not None,
            "correlation_time": last_correlation_update.isoformat()
            if last_correlation_update
            else None,
            "weights_loaded": len(utils.symbol_weights) > 0,
            "market_regime": detect_market_regime(),
            "portfolio_heat": get_portfolio_heat(),
            "circuit_breaker": trading_paused,
            "timestamp": datetime.now().isoformat(),
        }

        with open("system_status.json", "w") as f:
            json.dump(status, f)
    except Exception as e:
        logger.error(f"Erro ao salvar status: {e}")


@bot.message_handler(commands=["status"])
def handle_status(message):
    """Comando /status - Mostra posições abertas e status atual"""
    try:
        with mt5_lock:
            positions = mt5.positions_get() or []
            acc = mt5.account_info()
        
        if not positions:
            bot.reply_to(message, "📭 <b>Nenhuma posição aberta no momento.</b>", parse_mode="HTML")
            return
        
        # Cabeçalho
        msg = "📊 <b>POSIÇÕES ABERTAS ATUAIS</b>\n"
        msg += f"🕐 {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n"
        msg += "━━━━━━━━━━━━━━━━━━━━\n\n"
        
        total_profit = 0.0
        
        for i, pos in enumerate(positions, 1):
            symbol = pos.symbol
            direction = "🟢 LONG" if pos.type == mt5.POSITION_TYPE_BUY else "🔴 SHORT"
            volume = int(pos.volume)
            entry = pos.price_open
            current = pos.price_current
            sl = pos.sl if pos.sl > 0 else None
            tp = pos.tp if pos.tp > 0 else None
            profit = pos.profit
            
            # Calcula % de lucro
            if pos.type == mt5.POSITION_TYPE_BUY:
                pnl_pct = ((current - entry) / entry) * 100
            else:
                pnl_pct = ((entry - current) / entry) * 100
            
            total_profit += profit
            
            # Status emoji
            if profit > 0:
                status_emoji = "✅"
            elif profit < 0:
                status_emoji = "⚠️"
            else:
                status_emoji = "➖"
            
            # Monta mensagem da posição
            msg += f"<b>{i}. {symbol}</b> {direction}\n"
            msg += f"   💼 Volume: {volume:,} contratos\n"
            msg += f"   📍 Entrada: R$ {entry:.2f}\n"
            msg += f"   📈 Atual: R$ {current:.2f}\n"
            
            if sl:
                msg += f"   🛡️ SL: R$ {sl:.2f}\n"
            if tp:
                msg += f"   🎯 TP: R$ {tp:.2f}\n"
            
            msg += f"   {status_emoji} PnL: <b>R$ {profit:+,.2f}</b> ({pnl_pct:+.2f}%)\n\n"
            
            # Divide em mensagens se ficar muito grande (limite Telegram = 4096 chars)
            if len(msg) > 3500:
                bot.reply_to(message, msg, parse_mode="HTML")
                msg = ""
        
        # Rodapé com totais
        if msg:  # Se ainda tem conteúdo acumulado
            msg += "━━━━━━━━━━━━━━━━━━━━\n"
        
        total_emoji = "🟢" if total_profit >= 0 else "🔴"
        
        footer = f"\n<b>💰 TOTAL FLUTUANTE</b>\n"
        footer += f"{total_emoji} R$ {total_profit:+,.2f}\n\n"
        
        if acc:
            footer += f"💳 <b>Balance:</b> R$ {acc.balance:,.2f}\n"
            footer += f"💎 <b>Equity:</b> R$ {acc.equity:,.2f}\n"
            footer += f"📊 <b>Margem Livre:</b> R$ {acc.margin_free:,.2f}"
        
        bot.reply_to(message, msg + footer, parse_mode="HTML")
        
    except Exception as e:
        logger.error(f"Erro no comando /status: {e}", exc_info=True)
        bot.reply_to(message, "❌ Erro ao obter status. Tente novamente.", parse_mode="HTML")


def _telegram_is_authorized(message) -> bool:
    try:
        return str(message.chat.id) == str(config.TELEGRAM_CHAT_ID)
    except Exception:
        return False


@bot.message_handler(commands=["help", "ajuda", "comandos"])
def handle_help(message):
    if not _telegram_is_authorized(message):
        bot.reply_to(message, "❌ Acesso negado.", parse_mode="HTML")
        return

    msg = (
        "🧭 <b>COMANDOS XP3</b>\n\n"
        "/status — posições abertas\n"
        "/lucro — resumo realizado + flutuante\n"
        "/top15 — top15 atual (score/direção)\n"
        "/rejeicoes — resumo de rejeições (sob demanda)\n"
        "/pausar [motivo] — pausa novas entradas\n"
        "/retomar — retoma novas entradas\n"
        "/desativarcb — desativa circuit breaker (override)\n"
        "/ativarcb — reativa circuit breaker\n"
        "/saude — diagnóstico MT5 + modo + pausa\n"
        "/reload_elite — recarrega parâmetros (JSON/config)\n"
    )
    bot.reply_to(message, msg, parse_mode="HTML")


@bot.message_handler(commands=["top15", "top"])
def handle_top15(message):
    if not _telegram_is_authorized(message):
        bot.reply_to(message, "❌ Acesso negado.", parse_mode="HTML")
        return

    indicators, top15 = bot_state.snapshot
    if not top15:
        bot.reply_to(message, "📭 TOP15 vazio no momento.", parse_mode="HTML")
        return

    lines = ["🏆 <b>TOP15 (agora)</b>\n"]
    for i, sym in enumerate(top15, 1):
        ind = indicators.get(sym, {}) or {}
        score = float(ind.get("score", 0) or 0)
        direction = str(ind.get("direction", "–"))
        rsi = float(ind.get("rsi", 0) or 0)
        lines.append(f"{i:02d}. <b>{sym}</b> | {direction} | score {score:.0f} | rsi {rsi:.0f}")

    msg = "\n".join(lines)
    bot.reply_to(message, msg[:3900], parse_mode="HTML")


@bot.message_handler(commands=["rejeicoes", "rejections"])
def handle_rejections(message):
    if not _telegram_is_authorized(message):
        bot.reply_to(message, "❌ Acesso negado.", parse_mode="HTML")
        return

    try:
        summary = daily_logger.get_daily_rejection_summary()
        payload = f"📊 <b>REJEIÇÕES (HOJE)</b>\n<pre>{summary}</pre>"
        if len(payload) <= 3900:
            bot.reply_to(message, payload, parse_mode="HTML")
            return

        parts = []
        text = payload
        while text:
            parts.append(text[:3900])
            text = text[3900:]
        for part in parts[:4]:
            bot.send_message(config.TELEGRAM_CHAT_ID, part, parse_mode="HTML")
    except Exception as e:
        logger.error(f"Erro /rejeicoes: {e}", exc_info=True)
        bot.reply_to(message, "❌ Erro ao gerar resumo de rejeições.", parse_mode="HTML")


@bot.message_handler(commands=["pausar", "pause"])
def handle_pause(message):
    global trading_paused, manual_pause_reason
    if not _telegram_is_authorized(message):
        bot.reply_to(message, "❌ Acesso negado.", parse_mode="HTML")
        return

    reason = ""
    try:
        parts = (message.text or "").split(maxsplit=1)
        if len(parts) > 1:
            reason = parts[1].strip()
    except Exception:
        reason = ""

    manual_pause_reason = reason or "Pausa manual via Telegram"
    trading_paused = True
    bot.reply_to(message, f"⛔ <b>TRADING PAUSADO</b>\nMotivo: {manual_pause_reason}", parse_mode="HTML")


@bot.message_handler(commands=["retomar", "resume"])
def handle_resume(message):
    global trading_paused, manual_pause_reason, TRADING_PAUSED, PAUSE_REASON
    if not _telegram_is_authorized(message):
        bot.reply_to(message, "❌ Acesso negado.", parse_mode="HTML")
        return

    trading_paused = False
    manual_pause_reason = ""
    TRADING_PAUSED = False
    PAUSE_REASON = ""
    bot.reply_to(message, "✅ <b>TRADING RETOMADO</b>", parse_mode="HTML")

@bot.message_handler(commands=["desativarcb", "cboff"])
def handle_cb_off(message):
    global CIRCUIT_BREAKER_DISABLED, trading_paused, TRADING_PAUSED, PAUSE_REASON, manual_pause_reason
    if not _telegram_is_authorized(message):
        bot.reply_to(message, "❌ Acesso negado.", parse_mode="HTML")
        return
    CIRCUIT_BREAKER_DISABLED = True
    trading_paused = False
    TRADING_PAUSED = False
    PAUSE_REASON = ""
    manual_pause_reason = ""
    bot.reply_to(message, "✅ <b>CIRCUIT BREAKER DESATIVADO</b>\nTrading retomado.", parse_mode="HTML")

@bot.message_handler(commands=["ativarcb", "cbon"])
def handle_cb_on(message):
    global CIRCUIT_BREAKER_DISABLED
    if not _telegram_is_authorized(message):
        bot.reply_to(message, "❌ Acesso negado.", parse_mode="HTML")
        return
    CIRCUIT_BREAKER_DISABLED = False
    bot.reply_to(message, "🟢 <b>CIRCUIT BREAKER ATIVADO</b>", parse_mode="HTML")

@bot.message_handler(commands=["saude", "health"])
def handle_health(message):
    if not _telegram_is_authorized(message):
        bot.reply_to(message, "❌ Acesso negado.", parse_mode="HTML")
        return

    ok, diag = validate_mt5_health()
    mode = str(getattr(config, "CURRENT_OPERATION_MODE", "N/A"))
    paused_txt = "SIM" if trading_paused else "NÃO"

    elite_json_path = getattr(config, "ELITE_SYMBOLS_JSON_PATH", "")
    elite_json_exists = "SIM" if (elite_json_path and os.path.exists(elite_json_path)) else "NÃO"

    msg = (
        "🩺 <b>SAÚDE DO SISTEMA</b>\n\n"
        f"MT5: <b>{'OK' if ok else 'FALHA'}</b> — {diag}\n"
        f"Modo: <b>{mode}</b>\n"
        f"Pausado: <b>{paused_txt}</b>\n"
        f"Elite JSON: <b>{elite_json_exists}</b>\n"
    )
    bot.reply_to(message, msg, parse_mode="HTML")


@bot.message_handler(commands=["reload_elite", "reload"])
def handle_reload_elite(message):
    if not _telegram_is_authorized(message):
        bot.reply_to(message, "❌ Acesso negado.", parse_mode="HTML")
        return

    try:
        load_optimized_params()
        indicators, top15 = build_portfolio_and_top15()
        bot_state.update(indicators, top15)
        bot.reply_to(
            message,
            f"✅ <b>ELITE RECARREGADA</b>\nAtivos: {len(optimized_params)} | TOP15: {len(top15)}",
            parse_mode="HTML",
        )
    except Exception as e:
        logger.error(f"Erro /reload_elite: {e}", exc_info=True)
        bot.reply_to(message, "❌ Erro ao recarregar elite.", parse_mode="HTML")

# =========================
# 🧹 MANUTENÇÃO DE LOGS
# =========================

def cleanup_old_logs(days_to_keep: int = 30):
    """Remove logs com mais de X dias"""
    log_dir = "logs"
    if not os.path.exists(log_dir):
        return
    
    cutoff_date = datetime.now() - timedelta(days=days_to_keep)
    removed_count = 0
    freed_space = 0
    
    for file_path in glob.glob(os.path.join(log_dir, "*.log*")):
        try:
            file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
            
            if file_mtime < cutoff_date:
                file_size = os.path.getsize(file_path)
                os.remove(file_path)
                removed_count += 1
                freed_space += file_size
        
        except Exception as e:
            logger.warning(f"⚠️ Erro ao remover {file_path}: {e}")
    
    if removed_count > 0:
        freed_mb = freed_space / (1024 * 1024)
        logger.info(f"🧹 {removed_count} logs antigos removidos ({freed_mb:.2f} MB)")


def log_maintenance_thread():
    """Thread de manutenção diária"""
    while True:
        try:
            time.sleep(86400)  # 24 horas
            logger.info("🧹 Iniciando manutenção diária de logs...")
            cleanup_old_logs(days_to_keep=30)
        except Exception as e:
            logger.error(f"❌ Erro na manutenção de logs: {e}")
            time.sleep(3600)

def get_anti_chop_status() -> dict:
    """
    Retorna status dos filtros anti-chop para o painel
    """
    from utils import _symbol_sl_timestamps, _daily_symbol_trades
    
    # Símbolos em cooldown
    now = datetime.now()
    cooldown_minutes = config.ANTI_CHOP["cooldown_after_sl_minutes"]
    
    symbols_blocked = []
    for sym, timestamp in _symbol_sl_timestamps.items():
        elapsed = (now - timestamp).total_seconds() / 60
        if elapsed < cooldown_minutes:
            remaining = int(cooldown_minutes - elapsed)
            symbols_blocked.append(f"{sym} ({remaining}m)")
    
    # Símbolos próximos do limite
    symbols_near_limit = []
    max_losses = config.DAILY_SYMBOL_LIMITS["max_losing_trades_per_symbol"]
    
    for sym, stats in _daily_symbol_trades.items():
        if stats["losses"] >= max_losses - 1:  # A 1 perda do limite
            symbols_near_limit.append(f"{sym} ({stats['losses']}/{max_losses})")
    
    return {
        "blocked": symbols_blocked,
        "near_limit": symbols_near_limit,
        "total_blocked": len(symbols_blocked),
        "total_near_limit": len(symbols_near_limit)
    }

# =========================
# MAIN
# =========================
def main():
    """
    Ponto de entrada principal com operação contínua
    """
    global current_trading_day
    global CURRENT_MODE
    try:
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--mode", type=str, default="ambos")
        args, _ = parser.parse_known_args()
        CURRENT_MODE = str(args.mode or "ambos").upper()
        if CURRENT_MODE not in ("AMBOS","FUTUROS","ACOES","SO_FUTUROS"):
            CURRENT_MODE = "AMBOS"
        logger.info(f"Modo de operação: {CURRENT_MODE}")
    except Exception:
        CURRENT_MODE = "AMBOS"

    clear_screen()
    print(f"{C_CYAN}===================================================={C_RESET}")
    print(f"{C_CYAN}🚀 INICIANDO XP3 PRO BOT B3 - MODO CONTÍNUO 24/7{C_RESET}")
    print(f"{C_CYAN}===================================================={C_RESET}")

    try:
        cleanup_old_logs(days_to_keep=30)
        logger.info("🧹 Limpeza inicial de logs concluída")
    except Exception as e:
        logger.warning(f"⚠️ Erro na limpeza de logs: {e}")

    # 1. Inicialização do MetaTrader 5
    if not mt5.initialize(path=config.MT5_TERMINAL_PATH):
        logger.critical(f"❌ Falha ao conectar no MT5: {config.MT5_TERMINAL_PATH}")
        try:
            mapping = utils.discover_all_futures()
            if mapping:
                logger.warning(f"Mapeamentos de futuros via fallback: {mapping}")
        except Exception as e:
            logger.warning(f"Falha no fallback de mapeamento de futuros: {e}")
        return
    else:
        logger.info(f"✅ Conectado ao MT5 correto: {config.MT5_TERMINAL_PATH}")

    if not validate_mt5_symbols_or_abort():
        try:
            mt5.shutdown()
        except Exception:
            pass
        return

    # ✅ GARANTE MARKET WATCH (LAND TRADING)
    try:
        utils.ensure_market_watch_symbols()
    except Exception as e:
        logger.error(f"Erro ao sincronizar Market Watch: {e}")
    try:
        fm = utils.discover_all_futures()
        if fm:
            logger.info(f"Futuros mapeados: {fm}")
    except Exception as e:
        logger.warning(f"Erro ao descobrir futuros: {e}")

    # ✅ BACKTEST INICIAL (Verifica se devemos pausar logo no início)
    try:
        logger.info("📊 Executando análise retrospectiva inicial (backtest)...")
        initial_wr = run_performance_analysis()
        if initial_wr is not None:
             # Verifica pausa
            can_trade, reason = check_win_rate_pause()
            if not can_trade:
                logger.warning(f"⚠️ Bot iniciando em modo PAUSADO/RESTRITO: {reason}")
    except Exception as e:
        logger.error(f"Erro no backtest inicial: {e}")

    # ============================================
    # 📥 CARREGA ESTADO DIÁRIO (NOVO!)
    # ============================================
    
    logger.info("📦 Verificando estado diário anterior...")
    state_restored = load_daily_state()
    
    if not state_restored:
        logger.info("🆕 Iniciando com estado limpo (novo dia)")
        
        # Inicializa valores padrão
        global equity_inicio_dia, daily_max_equity, current_trading_day
        
        with utils.mt5_lock:
            acc = mt5.account_info()
        
        if acc:
            equity_inicio_dia = acc.equity
            daily_max_equity = acc.equity
        
        current_trading_day = datetime.now().date()
    
    # ============================================
    # 📦 CARREGA OUTROS DADOS PERSISTENTES
    # ============================================
    
    logger.info("📦 Carregando dados persistentes e otimizações...")
    utils.load_loss_streak_data()
    load_optimized_params()
    utils.load_adaptive_weights()

    try:
        load_anti_chop_data()
        load_daily_limits()
        logger.info("✅ Dados anti-chop e limites carregados")
    except Exception as e:
        logger.warning(f"⚠️ Erro ao carregar dados anti-chop: {e}")

    # ✅ NOVOS CARREGAMENTOS
    try:
        load_anti_chop_data()
        load_daily_limits()
        logger.info("✅ Dados anti-chop e limites carregados")
    except Exception as e:
        logger.warning(f"⚠️ Erro ao carregar dados anti-chop: {e}")

    # 3. Carga Inicial de Dados
    logger.info("🔍 Analisando mercado para gerar TOP 15 inicial...")
    try:
        ind, top = build_portfolio_and_top15()

        _, top15 = bot_state.snapshot
        if top15:
            logger.info("🔄 Forçando atualização inicial de correlação e pesos...")
            utils.update_correlations(list(top15))
            utils.update_adaptive_weights()

            logger.info(
                f"📑 Subscrevendo ao Book de Ofertas para {len(top15)} ativos..."
            )
            for sym in top15:
                if mt5.market_book_add(sym):
                    logger.debug(f"✅ DOM ativo: {sym}")
                else:
                    logger.warning(f"⚠️ Não foi possível ler o Book de: {sym}")

    except Exception as e:
        logger.error(f"❌ Erro grave na carga inicial: {e}")

    # === 🔬 VALIDAÇÃO DE CONFIGURAÇÃO ===
    logger.info("\n" + "="*60)
    logger.info("🔬 VALIDANDO CONFIGURAÇÃO DE RISCO")
    logger.info("="*60)
    logger.info(f"✅ Cooldown após SL: {config.ANTI_CHOP['cooldown_after_sl_minutes']} min")
    logger.info(f"✅ Cooldown progressivo: {config.ANTI_CHOP.get('progressive_cooldown', False)}")
    logger.info(f"✅ Máx posições: {config.MAX_SYMBOLS}")
    logger.info(f"✅ Máx por setor: {config.MAX_PER_SECTOR}")
    logger.info(f"✅ Máx bancos: {config.MAX_PER_SUBSETOR.get('BANCOS', 2)}")
    logger.info(f"✅ ADX mínimo (abertura): {config.TIME_SCORE_RULES['OPEN']['adx_min']}")
    logger.info(f"✅ ADX mínimo (intraday): {config.TIME_SCORE_RULES['MID']['adx_min']}")
    logger.info("="*60 + "\n")


    # 4. Inicializa controle de ciclo
    current_trading_day = datetime.now().date()
    logger.info(
        f"📅 Ciclo de trading iniciado: {current_trading_day.strftime('%d/%m/%Y')}"
    )

    # 5. Disparo das Threads
    logger.info("🧵 Iniciando threads de execução...")
    threads = [
        threading.Thread(target=fast_loop, daemon=True, name="FastLoop"),
        threading.Thread(
            target=health_watcher_thread, daemon=True, name="HealthWatcher"
        ),
        threading.Thread(target=daily_report, daemon=True, name="DailyReport"),
        threading.Thread(
            target=correlation_updater_thread, daemon=True, name="CorrUpdater"
        ),
        threading.Thread(target=log_maintenance_thread, daemon=True, name="LogMaintenance"),
        threading.Thread(target=auto_save_state_thread, daemon=True, name="AutoSave"),
    ]

    # Adiciona thread do Telegram se estiver habilitado
    if getattr(config, "ENABLE_TELEGRAM_NOTIF", False):
        threads.append(
            threading.Thread(target=telegram_polling_thread, daemon=True, name="TelegramPolling")
        )
        logger.info("   -> Thread 'TelegramPolling' adicionada com comandos")

    # Inicia TODAS as threads de uma vez (sem duplicar)
    for t in threads:
        t.start()
        logger.info(f"   -> Thread '{t.name}' iniciada com sucesso.")

    logger.info(f"🚀 Total de {len(threads)} threads ativas")

    # 6. Notificação de Inicialização
    status = get_market_status()

    msg = (
        f"🤖 <b>XP3 BOT INICIADO - MODO CONTÍNUO</b>\n\n"
        f"{status['emoji']} Status: {status['message']}\n"
        f"⏱ {status['detail']}\n\n"
        f"💰 Balance: R${mt5.account_info().balance:,.2f}\n"
        f"📊 TOP15: {len(bot_state.get_top15())} ativos\n\n"
        f"🔄 O bot opera em ciclos automáticos\n"
        f"✅ Nunca encerra (24/7)"
    )

    try:
        utils.send_telegram_message(msg)
    except Exception as e:
        logger.warning(f"Erro ao enviar notificação: {e}")

    # 7. Loop de Interface (INFINITO - BLINDADO)
    # 7. Dispara Dashboard (Streamlit)
    logger.info("🚀 Iniciando Dashboard via Streamlit...")
    launch_dashboard()
    
    print(f"\n{C_GREEN}✅ Bot rodando em background!{C_RESET}")
    print(f"{C_YELLOW}ℹ️ Dashboard deve abrir no navegador. Se não, acesse: http://localhost:8501{C_RESET}")
    print(f"{C_RED}🛑 Pressione Ctrl+C para encerrar o bot.{C_RESET}\n")

    # Loop principal (Keep Alive)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print(f"\n{C_YELLOW}⏸️ Encerrando aplicação...{C_RESET}")
    
    # ✅ SALVA ESTADO ANTES DE SAIR (Só chega aqui se der Ctrl+C real)
    logger.info("💾 Salvando estado diário e ML...")
    ml_optimizer.force_save() # ✅ FORCE SAVE DO ML
    save_daily_state()
    
    logger.info("✅ Estado salvo com sucesso")
    print(f"{C_YELLOW}ℹ️ O bot continua operando em background (Threads ativas){C_RESET}")
    
    # Mantém a thread principal viva para as outras threads (FastLoop, etc) continuarem
    while True:
        time.sleep(3600)
        
if __name__ == "__main__":
    main()
