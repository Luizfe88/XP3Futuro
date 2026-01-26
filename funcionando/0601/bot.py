import json
import pandas as pd
import time
import threading
import logging
from datetime import datetime, date, timedelta
from threading import Lock, RLock
from collections import deque, defaultdict, OrderedDict
import MetaTrader5 as mt5
import config
import utils
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
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
def get_market_status() -> dict:
    """
    Retorna status detalhado do mercado (VERSÃO CONTÍNUA)
    """
    now = datetime.now()
    current_time = now.time()
    today = now.date()

    # Configurações de horário
    start = datetime.strptime(config.TRADING_START, "%H:%M").time()
    no_entry = datetime.strptime(config.NO_ENTRY_AFTER, "%H:%M").time()
    force_close = datetime.strptime(config.CLOSE_ALL_BY, "%H:%M").time()

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
            "detail": f"Operando até {config.NO_ENTRY_AFTER}",
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
            "detail": f"Fechamento às {config.CLOSE_ALL_BY}",
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


# === New PositionManager (Priority 5) ===
@dataclass
class PositionStatus:
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
        close_by = datetime.strptime(self.config.CLOSE_ALL_BY, "%H:%M").time()
        if now.time() >= close_by:
            return "Day Close Forced"

        # 2. Time-stop por candles
        max_candles = self.config.MAX_TRADE_DURATION_CANDLES
        candles_open = int(pos.time_open_minutes / 15)

        if candles_open >= max_candles:
            return f"Time-stop ({candles_open} candles)"

        return None

    def should_apply_breakeven(
        self, pos: PositionStatus, atr: float
    ) -> Optional[float]:
        """Retorna novo SL para breakeven se aplicável"""
        if not self.config.ENABLE_BREAKEVEN:
            return None

        if pos.time_open_minutes < 5:
            return None

        if pos.profit_atr < 2.0:
            return None

        buffer = atr * 0.2

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
        """Retorna volume parcial se aplicável"""
        if not self.config.ENABLE_PARTIAL_CLOSE:
            return None

        if pos.time_open_minutes < 10:
            return None

        if "PARTIAL" in getattr(pos, "comment", ""):
            return None

        threshold = regime_config["partial_mult"] + 0.5

        if pos.profit_atr >= threshold:
            partial_volume = round(pos.volume * self.config.PARTIAL_PERCENT / 100) * 100
            if partial_volume >= 100:
                return partial_volume

        return None

    def calculate_trailing_sl(
        self, pos: PositionStatus, atr: float, regime_config: dict
    ) -> Optional[float]:
        """Retorna novo SL trailing se aplicável"""
        if not self.config.ENABLE_TRAILING_STOP:
            return None

        if pos.profit_atr < 2.5:
            return None

        trail_mult = (
            regime_config["trailing_tight"] + 0.3
            if pos.profit_atr >= 4.0
            else regime_config["trailing_initial"] + 0.5
        )

        if pos.side == "BUY":
            new_sl = pos.current_price - (atr * trail_mult)
            if new_sl <= pos.sl:
                return None
        else:
            new_sl = pos.current_price + (atr * trail_mult)
            if new_sl >= pos.sl:
                return None

        improvement = abs(new_sl - pos.sl) / atr

        if improvement >= 0.3:
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
        ind = indicators.get(symbol, {})

        if not ind or ind.get("error"):
            return None

        atr = ind.get("atr")
        if not atr or atr <= 0:
            return None

        # Monta status
        side = "BUY" if mt5_position.type == mt5.POSITION_TYPE_BUY else "SELL"
        profit_points = (
            mt5_position.price_current - mt5_position.price_open
            if side == "BUY"
            else mt5_position.price_open - mt5_position.price_current
        )

        pos = PositionStatus(
            ticket=mt5_position.ticket,
            symbol=symbol,
            side=side,
            entry_price=mt5_position.price_open,
            current_price=mt5_position.price_current,
            sl=mt5_position.sl,
            tp=mt5_position.tp,
            volume=mt5_position.volume,
            profit_atr=profit_points / atr,
            time_open_minutes=(
                time.time() - position_open_times.get(mt5_position.ticket, time.time())
            )
            / 60,
            regime=self._detect_regime(ind),
        )

        # 1️⃣ TIME-STOP
        close_reason = self.should_close_by_time(pos)
        if close_reason:
            return ("CLOSE", close_reason)

        # 2️⃣ BREAKEVEN
        new_sl_be = self.should_apply_breakeven(pos, atr)
        if new_sl_be:
            return ("MODIFY_SL", new_sl_be, "Breakeven")

        # 3️⃣ PARTIAL CLOSE
        regime_config = self.config.TP_RULES[pos.regime]
        partial_vol = self.should_partial_close(pos, regime_config)
        if partial_vol:
            return ("PARTIAL", partial_vol)

        # 4️⃣ TRAILING STOP
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
    ✅ NOVA VERSÃO: Gestão modular usando PositionManager
    Substitui manage_positions_advanced() após testes
    """
    manager = PositionManager(config)

    with utils.mt5_lock:
        positions = mt5.positions_get() or []

    if not is_valid_dataframe(positions):
        return

    # Obtém snapshot thread-safe dos indicadores
    indicators, _ = bot_state.snapshot

    for pos in positions:
        try:
            # Delega para PositionManager
            action = manager.manage_single_position(pos, indicators)

            if not action:
                continue

            # Executa ação retornada
            if action[0] == "CLOSE":
                with utils.mt5_lock:
                    tick = mt5.symbol_info_tick(pos.symbol)
                if not tick:
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
            logger.error(f"Erro ao gerenciar {pos.symbol}: {e}", exc_info=True)
            continue


def health_watcher_thread():
    while True:
        time.sleep(30)  # Verifica a cada 30s
        health_monitor.check_health()


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
alerts = deque(maxlen=10)
alerts_lock = Lock()
failure_lock = Lock()
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

def setup_logging():
    """Sistema profissional de logs com rotação automática"""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Handler principal (rotação diária)
    main_handler = TimedRotatingFileHandler(
        filename=os.path.join(log_dir, "xp3_bot.log"),
        when="midnight",
        interval=1,
        backupCount=30,  # Mantém 30 dias
        encoding="utf-8"
    )
    main_handler.setLevel(logging.INFO)
    main_handler.setFormatter(
        logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    )
    
    # Handler de erros (rotação semanal)
    error_handler = TimedRotatingFileHandler(
        filename=os.path.join(log_dir, "xp3_errors.log"),
        when="W0",  # Segunda-feira
        interval=1,
        backupCount=12,  # 12 semanas
        encoding="utf-8"
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(
        logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s')
    )
    
    # Console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(
        logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    )
    
    # Configura logger raiz
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.handlers.clear()
    root_logger.addHandler(main_handler)
    root_logger.addHandler(error_handler)
    root_logger.addHandler(console_handler)
    
    return logging.getLogger("bot")

# Chama setup
logger = setup_logging()

# =========================
# TIMEFRAMES
# =========================
TIMEFRAME_BASE = mt5.TIMEFRAME_M15
TIMEFRAME_MACRO = getattr(mt5, f"TIMEFRAME_{config.MACRO_TIMEFRAME}", mt5.TIMEFRAME_H1)

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
        equity_inicio_dia = state.get("equity_inicio_dia", 0.0)
        daily_max_equity = state.get("daily_max_equity", 0.0)
        
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

            # 2. Após o sucesso, aguarda 600 segundos (10 minutos)
            time.sleep(600)

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
            df = safe_copy_rates(sym, TIMEFRAME_BASE, 1000)
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
                indicators={"rsi": 50, "adx": 0, "spread_pips": 0, "volume_ratio": 0, "ema_trend": "N/A"}
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

        # Cálculo do SCORE
        score = 1
        if ind["ema_fast"] > ind["ema_slow"]:
            score += 50
        if 30 <= ind["rsi"] <= 70:
            score += 20

        # Calcula DIREÇÃO
        if ind["ema_fast"] > ind["ema_slow"] and score >= config.MIN_SIGNAL_SCORE:
            direction = "↑ LONG"
            signal = "BUY"
        elif ind["ema_fast"] < ind["ema_slow"] and score >= config.MIN_SIGNAL_SCORE:
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
        if score >= config.MIN_SIGNAL_SCORE - 10:  # ← 10 pontos de margem
            logger.info(f"🔥 {sym}: Score {score} (próximo de setup)")
        elif score < 20:
            logger.debug(f"⏸️ {sym}: Score {score} (muito fraco)")  # Debug level

        # ✅ LOG: Análise completa
        if signal == "NONE":
            daily_logger.log_analysis(
                symbol=sym,
                signal="NONE",
                strategy="ELITE",
                score=score,
                rejected=True,
                reason=f"📊 Score {score:.0f} < {config.MIN_SIGNAL_SCORE} (aguardando setup)",
                indicators={
                    "rsi": ind.get("rsi", 50),
                    "adx": ind.get("adx", 0),
                    "spread_pips": 0,  # B3 não usa spread em pips
                    "volume_ratio": ind.get("volume_ratio", 0),
                    "ema_trend": "UP" if ind["ema_fast"] > ind["ema_slow"] else "DOWN"
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

    gap = utils.get_open_gap(symbol, TIMEFRAME_BASE)
    if gap is not None and gap > config.MAX_GAP_OPEN_PCT:
        push_panel_alert(
            f"⚠️ {symbol} rejeitado: Gap de abertura alto ({gap:.2f}% > {config.MAX_GAP_OPEN_PCT * 100:.0f}%)",
            "INFO",
        )
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
            deviation = 30 if attempt <= 2 else 80 if attempt <= 4 else 150
            filling_type = mt5.ORDER_FILLING_RETURN

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
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": filling_type,
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
                final_profit_money = pos.profit  # Usa valor real do MT5
                final_pl_pct = (final_profit_money / (entry_price * volume)) * 100 if volume > 0 else 0
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
                save_trade(
                    symbol=symbol, side=side, volume=volume,
                    entry_price=entry_price, exit_price=final_exit_price,
                    sl=pos.sl, tp=pos.tp,
                    pnl_money=final_profit_money, pnl_pct=final_pl_pct,
                    reason=reason
                )
                log_trade_to_txt(
                    symbol=symbol, side=side, volume=volume,
                    entry_price=entry_price, exit_price=final_exit_price,
                    pnl_money=final_profit_money, pnl_pct=final_pl_pct,
                    reason=reason
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
                    ind_now = quick_indicators_custom(symbol, TIMEFRAME_BASE)
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

# ============================================
# CORREÇÃO: Variável Global para Cooldown de Entrada
# ============================================
last_entry_time = {}  # Adicione isso logo antes da função ou no topo do arquivo junto com as outras globais

def try_enter_position(symbol, side):
    """
    ✅ VERSÃO COM AUDITORIA: Registra motivo de cada rejeição
    """
    global last_entry_time
    
    # ========== VALIDAÇÕES COM LOG ==========
    
    # 1. Horário
    now = datetime.now().time()
    no_entry_time = datetime.strptime(config.NO_ENTRY_AFTER, "%H:%M").time()

    if now >= no_entry_time:
        # ✅ LOG
        daily_logger.log_analysis(
            symbol=symbol, signal=side, strategy="ELITE", score=0,
            rejected=True, reason="🕐 Horário: Sem novas entradas",
            indicators={"rsi": 0, "adx": 0, "spread_pips": 0, "volume_ratio": 0, "ema_trend": "N/A"}
        )
        return

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

    # 6. Cotação
    tick = mt5.symbol_info_tick(symbol)
    if not tick or tick.ask <= 0 or tick.bid <= 0:
        daily_logger.log_analysis(
            symbol=symbol, signal=side, strategy="ELITE", score=0,
            rejected=True, reason="❌ Sem cotação válida",
            indicators={"rsi": 0, "adx": 0, "spread_pips": 0, "volume_ratio": 0, "ema_trend": "N/A"}
        )
        return

    # 7. Indicadores
    ind_data = bot_state.get_indicators(symbol)

    if not ind_data or ind_data.get("error"):
        daily_logger.log_analysis(
            symbol=symbol, signal=side, strategy="ELITE", score=0,
            rejected=True, reason="❌ Erro nos indicadores",
            indicators={"rsi": 0, "adx": 0, "spread_pips": 0, "volume_ratio": 0, "ema_trend": "N/A"}
        )
        return

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
    existing_pos = mt5.positions_get(symbol=symbol)
    is_pyramiding = len(existing_pos) > 0
    
    if is_pyramiding:
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

    # ========== CÁLCULOS E VALIDAÇÕES FINAIS ==========
    
    entry_price = tick.ask if side == "BUY" else tick.bid
    atr_val = ind_data.get("atr", 0.10)

    if atr_val < (entry_price * 0.003):
        atr_val = entry_price * 0.005

    stop_dist = atr_val * 2.0
    base_vol = utils.calculate_position_size_atr(symbol, entry_price, stop_dist)
    volume = base_vol * 0.5 if is_pyramiding else base_vol

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

    sl, tp = utils.calculate_dynamic_sl_tp(symbol, side, entry_price, ind_data)

    if not utils.validate_order_params(symbol, volume, entry_price, sl, tp):
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
        
    if not utils.analyze_order_book_depth(symbol, side, volume):
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
        
    if not utils.is_spread_acceptable(symbol):
        daily_logger.log_analysis(
            symbol=symbol, signal=side, strategy="ELITE",
            score=ind_data.get("score", 0),
            rejected=True, reason="💸 Spread muito alto",
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
    order = validate_and_create_order(
        symbol=symbol, side=side, volume=volume, entry_price=entry_price, sl=sl, tp=tp
    )

    if not order:
        daily_logger.log_analysis(
            symbol=symbol, signal=side, strategy="ELITE",
            score=ind_data.get("score", 0),
            rejected=True, reason="❌ Validação falhou (R:R ou parâmetros)",
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
    
    pyr_count = 0

    if is_pyramiding:
        pos = existing_pos[0]
        pyr_count = pos.comment.count("PYR") if pos.comment else 0
    
    daily_trades_per_symbol[symbol] += 1
    comment = f"XP3_PYR_{pyr_count + 1}" if is_pyramiding else "XP3_INIT"

    logger.info(
        f"🚀 ENVIANDO {'PIRÂMIDE' if is_pyramiding else 'ENTRADA'} {side} em {symbol} | "
        f"Vol: {volume:.0f} @ {entry_price:.2f}"
    )

    request = order.to_mt5_request(comment=comment)
    result = mt5.order_send(request)

    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        last_entry_time[symbol] = time.time()

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
        with open("daily_equity.txt", "w") as f:
            f.write(str(equity_inicio_dia))
        trading_paused = False
        last_reset_day = today
        daily_trades_per_symbol.clear()
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
        # Calcula quantas posições fechar (30% = trava 70% do lucro indiretamente)
        with utils.mt5_lock:
            positions = mt5.positions_get() or []

        if len(positions) == 0:
            return

        # Fecha as posições mais arriscadas (menor lucro ou no prejuízo)
        positions_sorted = sorted(positions, key=lambda p: p.profit)

        # Fecha 30% das posições (arredonda pra cima)
        close_count = max(1, int(len(positions) * 0.3))

        for pos in positions_sorted[:close_count]:
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
                    reason="Profit Lock - Meta Diária Atingida",
                )

            except Exception as e:
                logger.error(f"Erro no Profit Lock para {pos.symbol}: {e}")

        push_alert(
            f"🎯 META DIÁRIA ATINGIDA! Lucro: R${daily_pnl:+.2f} ({daily_pnl_pct * 100:.1f}%) | "
            f"Fechadas {close_count} posições para proteção",
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

            # Aguarda 30 segundos
            time.sleep(30)

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
):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Se o preço de saída for 0, identificamos como uma ENTRADA
    tipo = "ENTRADA" if exit_price == 0 else "SAÍDA"

    # Formatamos a linha para incluir o tipo
    line = (
        f"{timestamp} | {tipo:<8} | {symbol:<6} | {side:<4} | "
        f"Vol: {volume:>7.0f} | Price: {entry_price:>6.2f} | "
        f"P&L: {pnl_money:>+8.2f} R$ ({pnl_pct:+.2f}%) | "
        f"Motivo: {reason}\n"
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
# PAINEL (VERSÃO RICH - ESTILO FOREX)
# ============================================
def render_panel_enhanced():
    """
    Painel visual estilo Forex Bot (Rich) com dados do B3 Bot.
    """
    # Se não tiver Rich, roda a versão antiga ou sai
    if console is None or Live is None:
        return

    # Suprime logs durante a renderização para não quebrar o layout
    root_logger = logging.getLogger()
    original_levels = []
    for handler in root_logger.handlers:
        original_levels.append(handler.level)
        handler.setLevel(logging.CRITICAL)

    def generate_display() -> Layout:
        # 1. COLETA DE DADOS
        market_status = get_market_status()
        current_indicators, top15_symbols = bot_state.snapshot
        
        with utils.mt5_lock:
            acc = mt5.account_info()
            positions = mt5.positions_get() or []

        now_str = datetime.now().strftime("%d/%m %H:%M:%S")

        # Fallback se MT5 cair
        if not acc:
            return Layout(Panel(Text("❌ Aguardando conexão MT5...", justify="center", style="bold red"), title="XP3 PRO B3", border_style="red"))

        # Cálculos Financeiros
        pnl_aberto = acc.equity - acc.balance
        pnl_color = "green" if pnl_aberto >= 0 else "red"
        
        daily_pnl = acc.equity - equity_inicio_dia
        daily_pnl_pct = (daily_pnl / equity_inicio_dia * 100) if equity_inicio_dia > 0 else 0
        daily_pnl_color = "green" if daily_pnl >= 0 else "red"
        
        daily_target = config.PROFIT_TARGETS["daily"]["target_return"] * 100
        daily_progress = min((daily_pnl_pct / daily_target) * 100, 100) if daily_target > 0 else 0

        # === LAYOUT PRINCIPAL ===
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=6),
            Layout(name="upper_body", size=10),
            Layout(name="positions", ratio=1),
            Layout(name="scanner", ratio=2),
            Layout(name="footer", size=3)
        )

        layout["upper_body"].split_row(
            Layout(name="system_status"),
            Layout(name="alerts")
        )

        # === 1. HEADER (Fixo) ===
        header_text = Text()
        header_text.append("🚀 XP3 PRO BOT - B3 [MODO CONTÍNUO]\n", style="bold cyan")
        
        # Status do Mercado
        if trading_paused:
             header_text.append("🚨 PAUSADO (CIRCUIT BREAKER) ", style="bold red")
        else:
            status_style = "bold green" if market_status['status'] == 'OPEN' else "bold yellow" if market_status['status'] == 'PRE_MARKET' else "bold red"
            header_text.append(f"{market_status['emoji']} {market_status['message']} ", style=status_style)
            if market_status['countdown']:
                header_text.append(f"({market_status['countdown']})", style="dim white")
        
        header_text.append("\n")
        
        # Financeiro
        header_text.append(f"💰 Balance: R${acc.balance:,.2f} | ", style="white")
        header_text.append(f"Equity: R${acc.equity:,.2f} | ", style="white")
        header_text.append(f"P/L Aberto: R${pnl_aberto:+,.2f} ", style=f"bold {pnl_color}")
        
        # Meta Diária
        meta_color = "green" if daily_progress >= 100 else "yellow"
        header_text.append(f"\n🎯 Meta Diária: ", style="white")
        header_text.append(f"{daily_progress:.1f}% ", style=f"bold {meta_color}")
        header_text.append(f"(R${daily_pnl:+,.2f} / {daily_pnl_pct:+.2f}%)", style=f"{daily_pnl_color}")
        header_text.append(f" | 🕐 {now_str}", style="dim right")

        header_panel = Panel(header_text, border_style="cyan")

        # === 2. SYSTEM STATUS (Esquerda Superior) ===
        sys_table = Table(box=None, show_header=False, expand=True)
        sys_table.add_column("Key", style="yellow")
        sys_table.add_column("Value", style="white")

        # Dados do sistema
        heat = get_portfolio_heat()
        heat_color = "red" if heat > 0.7 else "yellow" if heat > 0.5 else "green"
        
        anti_chop = get_anti_chop_status()
        anti_chop_str = f"🚫 {anti_chop['total_blocked']} Bloq" if anti_chop['total_blocked'] > 0 else "✅ Livre"

        lucro_realizado, qtd_trades = utils.calcular_lucro_realizado_txt()

        sys_table.add_row("🔥 Heat / Risco:", f"[{heat_color}]{heat:.3f}[/] / {utils.get_current_risk_pct()*100:.2f}%")
        sys_table.add_row("🌊 Regime:", f"{detect_market_regime()}")
        sys_table.add_row("⚡ Power Hour:", "🔥 ATIVA" if utils.is_power_hour() else "—")
        sys_table.add_row("🛡️ Anti-Chop:", anti_chop_str)
        sys_table.add_row("💰 Realizado Hoje:", f"R${lucro_realizado:,.2f} ({qtd_trades} trades)")

        status_panel = Panel(sys_table, title="⚙️ STATUS DO SISTEMA", border_style="blue")

        # === 3. ALERTS (Direita Superior) ===
        alert_text = Text()
        with alerts_lock:
            recent_alerts = list(alerts)[:6]
        
        if not recent_alerts:
            alert_text.append("(Nenhum alerta recente)", style="dim")
        else:
            for lvl, msg in recent_alerts:
                color = "red" if lvl == "CRITICAL" else "yellow" if lvl == "WARNING" else "white"
                # Remove timestamp duplicado se já houver na string
                clean_msg = msg.split("] ")[-1] if "] " in msg else msg
                time_alert = msg.split("] ")[0].replace("[", "") if "] " in msg else ""
                alert_text.append(f"{time_alert} ", style="dim cyan")
                alert_text.append(f"{clean_msg[:50]}\n", style=color)

        alert_panel = Panel(alert_text, title="🚨 ÚLTIMOS ALERTAS", border_style="yellow")

        # === 4. POSIÇÕES (Meio) ===
        pos_table = Table(show_header=True, header_style="bold magenta", border_style="blue", expand=True, box=box.SIMPLE_HEAD)
        pos_table.add_column("Sym", style="cyan", width=6)
        pos_table.add_column("Lado", width=6)
        pos_table.add_column("Vol", justify="right", width=6)
        pos_table.add_column("Entrada", justify="right", width=9)
        pos_table.add_column("Atual", justify="right", width=9)
        pos_table.add_column("P/L R$", justify="right", width=10)
        pos_table.add_column("%", justify="right", width=7)
        pos_table.add_column("Status", width=12)

        if positions:
            for pos in positions:
                tick = mt5.symbol_info_tick(pos.symbol)
                current_price = tick.bid if pos.type == mt5.POSITION_TYPE_BUY else tick.ask if tick else pos.price_current
                
                side = "BUY" if pos.type == mt5.POSITION_TYPE_BUY else "SELL"
                side_style = "green" if side == "BUY" else "red"
                
                profit = pos.profit
                profit_style = "green" if profit >= 0 else "red"
                
                pct = ((current_price - pos.price_open) / pos.price_open * 100) if side == "BUY" else ((pos.price_open - current_price) / pos.price_open * 100)
                
                # Status textual
                ind = current_indicators.get(pos.symbol, {})
                atr = ind.get("atr", 0.01)
                profit_atr = abs(current_price - pos.price_open) / atr if atr > 0 else 0
                
                status_txt = "Trailing" if profit_atr >= 2.5 else "Breakeven" if profit_atr >= 1.0 else "Normal"

                pos_table.add_row(
                    pos.symbol,
                    f"[{side_style}]{side}[/]",
                    f"{pos.volume:.0f}",
                    f"{pos.price_open:.2f}",
                    f"{current_price:.2f}",
                    f"[{profit_style}]{profit:+.2f}[/]",
                    f"[{profit_style}]{pct:+.1f}%[/]",
                    status_txt
                )
        else:
            pos_table.add_row("-", "-", "-", "-", "-", "-", "-", "[dim]Nenhuma posição aberta[/dim]")

        pos_panel = Panel(pos_table, title=f"💼 CARTEIRA ({len(positions)}/{config.MAX_SYMBOLS})", border_style="green")

        # === 5. TOP 15 SCANNER (Baixo) ===
        scan_table = Table(show_header=True, header_style="bold yellow", border_style="magenta", expand=True, box=box.SIMPLE_HEAD)
        scan_table.add_column("RK", width=3, justify="right")
        scan_table.add_column("Sym", style="cyan", width=6)
        scan_table.add_column("Score", width=5, justify="right")
        scan_table.add_column("Dir", width=4, justify="center")
        scan_table.add_column("RSI", width=4, justify="right")
        scan_table.add_column("ATR%", width=5, justify="right")
        scan_table.add_column("Preço", width=8, justify="right")
        scan_table.add_column("Setor", width=15)
        scan_table.add_column("Status", width=10)
        scan_table.add_column("Motivo", width=20)

        # Lógica de exibição do Top 15 (cópia da lógica original)
        positions_symbols = [p.symbol for p in positions]
        current_sectors = get_sector_counts()

        if top15_symbols:
            for rank, sym in enumerate(top15_symbols, 1):
                ind = current_indicators.get(sym, {})
                if not ind: continue

                score = utils.calculate_signal_score(ind)
                
                # Direção visual
                is_long = ind.get("ema_fast", 0) > ind.get("ema_slow", 0)
                dir_arrow = "↑" if is_long else "↓"
                dir_color = "green" if is_long else "red"
                
                # Cor do Score
                score_color = "bold green" if score >= 80 else "green" if score >= 60 else "yellow" if score >= 40 else "red"
                
                # Preço e Setor
                price = ind.get('close', 0)
                sector_raw = config.SECTOR_MAP.get(sym, "UNKNOWN")
                sector = sector_raw[:14]

                # Lógica de Status (Idêntica ao bot.py original)
                status_display = ""
                motive = ""
                status_color = ""

                if sym in positions_symbols:
                    status_display = "ABERTO"
                    status_color = "green"
                    motive = "Posição ativa"
                elif not market_status["new_entries_allowed"]:
                    status_display = "FECHADO"
                    status_color = "dim"
                    motive = "Horário"
                elif not additional_filters_ok(sym):
                    status_display = "FILTRO"
                    status_color = "yellow"
                    motive = "Vol/Gap/Spread"
                elif not macro_trend_ok(sym, "BUY" if is_long else "SELL"):
                    status_display = "MACRO"
                    status_color = "red"
                    motive = "Contra H1"
                elif current_sectors.get(sector_raw, 0) >= config.MAX_PER_SECTOR:
                    status_display = "SETOR"
                    status_color = "magenta"
                    motive = "Limite Setor"
                elif score < config.MIN_SIGNAL_SCORE:
                    status_display = "SCORE"
                    status_color = "dim"
                    motive = f"Score {score:.0f}"
                else:
                    status_display = "PRONTO"
                    status_color = "bold green blink"
                    motive = "Sinal Forte"

                scan_table.add_row(
                    str(rank),
                    sym,
                    f"[{score_color}]{score:.0f}[/]",
                    f"[{dir_color}]{dir_arrow}[/]",
                    f"{ind.get('rsi', 0):.0f}",
                    f"{min(ind.get('atr_real', 0)*5.3, 9.9):.1f}",
                    f"{price:.2f}",
                    sector,
                    f"[{status_color}]{status_display}[/]",
                    motive
                )
        else:
             scan_table.add_row("-", "-", "-", "-", "-", "-", "-", "-", "-", "Top 15 não carregado")

        scan_panel = Panel(scan_table, title="📊 SCANNER TOP 15 ELITE", border_style="yellow")

        # === 6. FOOTER (ML e info) ===
        ml_stats = f"🤖 ML Trades: {len(ml_optimizer.history) if hasattr(ml_optimizer, 'history') else 0} | Epsilon: {getattr(ml_optimizer, 'epsilon', 0):.3f}"
        footer_text = Text(f"{ml_stats} | 💾 Correlação: {'OK' if last_correlation_update else 'Pend'} | 📅 Ciclo: {current_trading_day}", style="dim", justify="center")
        footer_panel = Panel(footer_text, border_style="dim")

        # === MONTAGEM DO LAYOUT ===
        layout["header"].update(header_panel)
        layout["system_status"].update(status_panel)
        layout["alerts"].update(alert_panel)
        layout["positions"].update(pos_panel)
        layout["scanner"].update(scan_panel)
        layout["footer"].update(footer_panel)

        return layout

    # === LOOP DO LIVE RENDER ===
    try:
        with Live(
            generate_display(), 
            console=console, 
            screen=True, 
            refresh_per_second=1,
            auto_refresh=False  # Controle manual
        ) as live:
            while True:
                live.update(generate_display(), refresh=True)
                time.sleep(1)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.error(f"Erro no painel Rich: {e}")
    finally:
        # Restaura logs ao sair
        for handler, original_level in zip(root_logger.handlers, original_levels):
            handler.setLevel(original_level)


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
            close_time = datetime.strptime(config.CLOSE_ALL_BY, "%H:%M").time()

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

            # ============================================
            # 1️⃣ OBTÉM STATUS DO MERCADO
            # ============================================
            market_status = get_market_status()

            # ============================================
            # 2️⃣ ATUALIZA DADOS (SEMPRE)
            # ============================================
            new_indicators, new_top15 = build_portfolio_and_top15()
            update_bot_bridge()

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
                symbols_to_scan = bot_state.get_top15()

                for sym in symbols_to_scan:
                    ind_data = bot_state.get_indicators(sym)

                    if not ind_data or ind_data.get("error"):
                        continue

                    score = utils.calculate_signal_score(ind_data)

                    if score >= config.MIN_SIGNAL_SCORE:
                        side = (
                            "BUY"
                            if ind_data["ema_fast"] > ind_data["ema_slow"]
                            else "SELL"
                        )
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
            try:
                utils.save_loss_streak_data()
                utils.save_adaptive_weights()
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


@bot.message_handler(commands=["lucro", "status"])
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
    Thread para polling contínuo do bot Telegram com retry em caso de erro.
    """
    while True:
        try:
            bot.polling(none_stop=True, interval=0, timeout=20)
        except Exception as e:
            logger.error(f"Erro no polling Telegram: {e}")
            time.sleep(10)  # Aguarda 10s antes de retry

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
        return
    else:
        logger.info(f"✅ Conectado ao MT5 correto: {config.MT5_TERMINAL_PATH}")

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
        telegram_thread = threading.Thread(
            target=telegram_polling_thread, daemon=True, name="TelegramPolling"
        )
        threads.append(telegram_thread)
        logger.info(
            "   -> Thread 'TelegramPolling' adicionada (/lucro, /status ativados)"
        )

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

    # 7. Loop de Interface (INFINITO)
    logger.info("🖥️ Painel de controle ativo")

    # Chama o painel. Ele assume o controle da thread principal.
    # O loop infinito está DENTRO dele agora.
    render_panel_enhanced()

    # Se saiu do painel (Ctrl+C), executa o encerramento seguro
    print(f"\n{C_YELLOW}⏸️ Painel interrompido pelo usuário{C_RESET}")
    
    # ✅ SALVA ESTADO ANTES DE SAIR
    logger.info("💾 Salvando estado diário...")
    save_daily_state()
    
    # Salva outros dados
    utils.save_loss_streak_data()
    utils.save_adaptive_weights()
    save_anti_chop_data()
    save_daily_limits()
    
    logger.info("✅ Estado salvo com sucesso")
    print(f"{C_YELLOW}ℹ️ O bot continua operando em background (Threads ativas){C_RESET}")
    
    # Mantém a thread principal viva para as outras threads (FastLoop, etc) continuarem
    while True:
        time.sleep(3600)
        
if __name__ == "__main__":
    main()
