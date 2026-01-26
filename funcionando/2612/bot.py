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
)
from ml_optimizer import EnsembleOptimizer
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

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
    Gerencia o ciclo diário do bot sem encerrar o processo
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

        # Reset de variáveis diárias
        current_trading_day = today
        daily_cycle_completed = False
        daily_report_sent = False

        # Reset do circuit breaker
        with utils.mt5_lock:
            acc = mt5.account_info()

        if acc:
            equity_inicio_dia = acc.equity
            daily_max_equity = acc.equity

            # Salva para referência
            with open("daily_equity.txt", "w") as f:
                f.write(str(equity_inicio_dia))

            logger.info(f"💰 Equity inicial do dia: R${equity_inicio_dia:,.2f}")

        # Reset de contadores
        last_reset_day = today
        daily_trades_per_symbol.clear()

        push_alert(
            f"🌅 Novo ciclo de trading iniciado: {today.strftime('%d/%m/%Y')}", "INFO"
        )

    # ============================================
    # 2️⃣ HORÁRIO DE FECHAMENTO ATINGIDO
    # ============================================
    if market_status["should_close_positions"] and not daily_cycle_completed:
        logger.info("🔒 Horário de fechamento atingido - Encerrando ciclo diário")

        # Fecha todas as posições
        with utils.mt5_lock:
            positions = mt5.positions_get() or []

        if positions:
            logger.info(f"📊 Fechando {len(positions)} posições abertas...")
            close_all_positions(reason="Fechamento automático EOD")

            # Aguarda confirmação
            time.sleep(2)

            # Verifica se realmente fechou
            with utils.mt5_lock:
                remaining = mt5.positions_get() or []

            if remaining:
                logger.warning(f"⚠️ {len(remaining)} posições não fecharam!")
                push_alert(
                    f"⚠️ {len(remaining)} posições abertas após EOD",
                    "WARNING",
                    sound=True,
                )
            else:
                logger.info("✅ Todas as posições fechadas com sucesso")
        else:
            logger.info("ℹ️ Nenhuma posição aberta no fechamento")

        # Marca ciclo como completo
        daily_cycle_completed = True

        push_alert("✅ Ciclo diário concluído - Aguardando próximo pregão", "INFO")
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

    if not positions:
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
bot_state = BotState()  # Priority 1: Unified state
position_open_times = TimedCache(max_age_seconds=86400)  # Priority 2: 24h for positions
last_close_time = TimedCache(max_age_seconds=7200)  # Priority 2: 2h for close times

# =========================
# LOG
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("xp3_bot.log", encoding="utf-8"),
    ],
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

    if len(symbols) < 2:
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


# =========================
# BUILD TOP15
# =========================
def build_portfolio_and_top15():
    global _first_build_done
    scored = []
    indicators = {}

    elite_symbols = list(optimized_params.keys())

    if not elite_symbols:
        logger.error("❌ ELITE_SYMBOLS está vazio!")
        return {}, []

    # Mensagem de carregamento apenas na primeira vez
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
            ind = utils.get_cached_indicators(sym, TIMEFRAME_BASE, 300)
            if ind.get("error"):
                df = utils.safe_copy_rates(sym, TIMEFRAME_BASE, 300)
                # ind = utils.quick_indicators_custom(sym, TIMEFRAME_BASE, df=df, params=params)

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

    bot_state.update(indicators, selected_top)  # Atomic update
    

# Supondo que você já calculou a porcentagem e o status antes desta parte
    dados_para_compartilhar = {
        "top15": bot_state.snapshot[1], # Certifique-se que o snapshot[1] já tenha as 8 colunas
        "indicators": bot_state.snapshot[0],
        "last_update": datetime.now().strftime("%H:%M:%S")
    }

    # Caminho correto para gravação segura
    temp_file = "bot_bridge.json.tmp"
    final_file = "bot_bridge.json"

    try:
        with open(temp_file, "w", encoding="utf-8") as f:
            # O segredo está no cls=NpEncoder
            json.dump(dados_para_compartilhar, f, cls=NpEncoder, indent=4)
        
        # Substituição atômica para evitar que o dashboard leia um arquivo vazio
        os.replace(temp_file, final_file)
    except Exception as e:
        print(f"Erro ao salvar JSON: {e}")
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


def close_position(
    symbol: str, ticket: int, volume: float, price: float, reason: str = ""
):
    """
    ✅ VERSÃO CORRIGIDA: Registra logs/ML APENAS após confirmação do MT5
    - Previne P&L zerado
    - Previne duplicação de registros
    - Retry automático em falhas
    """
    # ============================================
    # 1️⃣ VALIDAÇÃO: Posição ainda existe?
    # ============================================
    with utils.mt5_lock:
        pos = mt5.positions_get(ticket=ticket)
        if not pos:
            logger.warning(f"⚠️ Ticket {ticket} não encontrado (já fechado?)")
            return False
        pos = pos[0]

    # ============================================
    # 2️⃣ CALCULA MÉTRICAS (antes de fechar)
    # ============================================
    side = "BUY" if pos.type == mt5.POSITION_TYPE_BUY else "SELL"
    entry = pos.price_open

    profit_loss_points = (price - entry) if side == "BUY" else (entry - price)
    profit_loss_money = profit_loss_points * volume
    pl_pct = (profit_loss_money / (entry * volume)) * 100 if volume > 0 else 0

    # Duração
    with position_open_times_lock:
        open_time = position_open_times.get(ticket)

    if open_time:
        duration_seconds = time.time() - open_time
        duration_str = f"{duration_seconds / 60:.1f} min"
    else:
        duration_str = "desconhecido"

    # ============================================
    # 3️⃣ PREPARA REQUEST
    # ============================================
    request = {
    "action": mt5.TRADE_ACTION_DEAL,
    "symbol": pos.symbol,
    "position": pos.ticket,
    "volume": pos.volume,
    "type": mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY,
    "price": price,
    "deviation": 20,
    "type_time": mt5.ORDER_TIME_GTC,
    "comment": f"XP3 CloseAll - {reason}",
}


    # ============================================
    # 4️⃣ ENVIA ORDEM COM RETRY
    # ============================================
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        with utils.mt5_lock:
            result = mt5.order_send(request)

        # ✅ SUCESSO - AGORA SIM REGISTRA TUDO!
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(
                f"✅ FECHADO (tentativa {attempt}): {symbol} {side} | "
                f"Vol: {volume:.0f} | P&L: R${profit_loss_money:+.2f} ({pl_pct:+.2f}%) | "
                f"Duração: {duration_str} | Motivo: {reason}"
            )

            # ============================================
            # 5️⃣ REGISTROS (SÓ APÓS SUCESSO!)
            # ============================================

            # 5.1 - Salva no banco
            try:
                save_trade(
                    symbol=symbol,
                    side=side,
                    volume=volume,
                    entry_price=entry,
                    exit_price=price,
                    sl=pos.sl,
                    tp=pos.tp,
                    pnl_money=profit_loss_money,
                    pnl_pct=pl_pct,
                    reason=reason,
                    ml_reward=0.0,  # Opcional: calcular reward ML aqui
                )
                logger.debug(f"💾 Trade salvo no banco: {symbol}")
            except Exception as e:
                logger.error(f"❌ Erro ao salvar no banco: {e}")

            # 5.2 - Salva no TXT
            try:
                log_trade_to_txt(
                    symbol=symbol,
                    side=side,
                    volume=volume,
                    entry_price=entry,
                    exit_price=price,
                    pnl_money=profit_loss_money,
                    pnl_pct=pl_pct,
                    reason=reason,
                )
                logger.debug(f"📝 Trade salvo no TXT: {symbol}")
            except Exception as e:
                logger.error(f"❌ Erro ao salvar no TXT: {e}")

            # 5.3 - Registra no ML
            try:
                utils.record_trade_outcome(symbol, profit_loss_money)
                
                with entry_indicators_lock:
                    ind_at_entry = entry_indicators.get(symbol)

                if ind_at_entry:
                    ml_optimizer.record_trade(
                        symbol=symbol, 
                        pnl_pct=pl_pct / 100, 
                        indicators=ind_at_entry
                    )
                    logger.debug(f"🧠 ML atualizado: {symbol}")
                    
                    # Remove indicadores salvos
                    entry_indicators.pop(symbol, None)
                    
            except Exception as e:
                logger.error(f"❌ Erro no ML para {symbol}: {e}")

            # 5.4 - Atualiza caches
            try:
                last_close_time.set(symbol, time.time())
                position_open_times.pop(ticket, None)
            except Exception as e:
                logger.error(f"❌ Erro ao atualizar cache: {e}")

            # 5.5 - Notificação
            pl_color = "🟢" if profit_loss_money > 0 else "🔴"
            push_alert(
                f"{pl_color} FECHADO {side} {symbol} | "
                f"{volume:.0f} ações | "
                f"P&L: R${profit_loss_money:+.2f} ({pl_pct:+.2f}%) | "
                f"{reason}"
            )

            # 5.6 - Telegram
            try:
                send_telegram_exit(
                    symbol=symbol,
                    side=side,
                    volume=volume,
                    entry_price=entry,
                    exit_price=price,
                    profit_loss=profit_loss_money,
                    reason=reason,
                )
                logger.debug(f"📱 Notificação Telegram enviada: {symbol}")
            except Exception as e:
                logger.warning(f"⚠️ Erro Telegram saída: {e}")

            return True

        # ⚠️ ERROS RECUPERÁVEIS (retry)
        elif result and result.retcode in [
            mt5.TRADE_RETCODE_REQUOTE,
            mt5.TRADE_RETCODE_PRICE_OFF,
            mt5.TRADE_RETCODE_TIMEOUT,
            mt5.TRADE_RETCODE_PRICE_CHANGED,
        ]:
            logger.warning(
                f"⚠️ Tentativa {attempt}/{max_retries} falhou: "
                f"{result.comment} (retcode {result.retcode})"
            )

            if attempt < max_retries:
                time.sleep(0.5)  # Aguarda 500ms

                # Atualiza preço para nova tentativa
                with utils.mt5_lock:
                    tick = mt5.symbol_info_tick(symbol)
                if tick:
                    request["price"] = tick.bid if side == "BUY" else tick.ask
                continue

        # 🚨 ERRO FATAL
        else:
            error_msg = result.comment if result else "Sem resposta do MT5"
            logger.error(
                f"❌ FALHA CRÍTICA ao fechar {symbol} | "
                f"Tentativa {attempt}/{max_retries} | "
                f"Erro: {error_msg} | "
                f"Retcode: {result.retcode if result else 'N/A'}"
            )

            if attempt < max_retries:
                time.sleep(1)  # Aguarda 1s em erro grave
                continue
            else:
                push_alert(
                    f"🚨 FALHA AO FECHAR {symbol} após {max_retries} tentativas: {error_msg}",
                    "CRITICAL",
                )
                return False

    # Se chegou aqui, falhou todas as tentativas
    logger.critical(f"🚨 Fechamento de {symbol} FALHOU após {max_retries} tentativas")
    return False


def close_all_positions(reason: str = "Fechamento diário"):
    """
    Fecha todas as posições SEM encerrar o bot
    Versão blindada MT5 com correção do filling_mode
    """
    logger.info(f"🔒 Iniciando fechamento global: {reason}")

    with utils.mt5_lock:
        positions = mt5.positions_get()

    if not positions:
        logger.info("ℹ️ Nenhuma posição aberta para fechar")
        return

    success = 0
    failed = []

    for pos in positions:
        try:
            with utils.mt5_lock:
                tick = mt5.symbol_info_tick(pos.symbol)
                symbol_info = mt5.symbol_info(pos.symbol)

            if not tick or not symbol_info:
                logger.error(f"❌ {pos.symbol}: sem tick ou symbol_info")
                failed.append(pos.ticket)
                continue

            # 🔒 Preço correto
            price = tick.bid if pos.type == mt5.POSITION_TYPE_BUY else tick.ask

            # ✅ CORREÇÃO: Atributo correto é filling_mode
            # Usa getattr com fallback para ORDER_FILLING_IOC
            filling = getattr(symbol_info, 'filling_mode', mt5.ORDER_FILLING_IOC)
            
            # Se filling_mode retornar 0 ou None, usa IOC como padrão
            if not filling or filling == 0:
                filling = mt5.ORDER_FILLING_IOC

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": pos.symbol,
                "position": pos.ticket,
                "volume": pos.volume,
                "type": mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                "price": price,
                "deviation": 20,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": filling,  # ✅ Agora usa o valor correto
                "comment": f"XP3 CloseAll - {reason}",
            }

            with utils.mt5_lock:
                result = mt5.order_send(request)

            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                success += 1
                logger.info(f"✅ Fechado {pos.symbol} | ticket {pos.ticket}")
            else:
                failed.append(pos.ticket)
                error_msg = result.comment if result else 'sem resposta'
                retcode = result.retcode if result else 'None'
                logger.error(
                    f"❌ Falha ao fechar {pos.symbol} | "
                    f"retcode={retcode} | msg={error_msg}"
                )

        except Exception as e:
            logger.error(f"🚨 Erro crítico ao fechar {pos.symbol}: {e}", exc_info=True)
            failed.append(pos.ticket)

    # 📊 Relatório final
    total = len(positions)

    if success == total:
        push_alert(f"✅ TODAS FECHADAS ({success}/{total}) | {reason}", "INFO")
    else:
        push_alert(
            f"⚠️ PARCIAL ({success}/{total}) | Falhas: {len(failed)}",
            "WARNING",
            sound=True,
        )

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
    Tenta entrar ou aumentar posição (pirâmide) com logs detalhados e registro para ML.
    Inclui proteções: cooldown DE ENTRADA, cooldown de saída, loss streak.
    """
    global last_entry_time # Necessário para modificar o dicionário global
    
    # =========================================
    # 1. VALIDAÇÃO: HORÁRIO DE ENTRADA
    # =========================================
    now = datetime.now().time()
    no_entry_time = datetime.strptime(config.NO_ENTRY_AFTER, "%H:%M").time()

    if now >= no_entry_time:
        return

    # =========================================
    # 2. COOLDOWN DE SAÍDA (30 minutos após última saída)
    # =========================================
    if time.time() - last_close_time.get(symbol, 0) < 1800:
        return

    # =========================================
    # 🆕 2.1 COOLDOWN DE ENTRADA (EVITA METRALHADORA)
    # =========================================
    # Só permite nova entrada/pirâmide se passaram 5 minutos (300s) da última entrada
    if time.time() - last_entry_time.get(symbol, 0) < 300:
        logger.debug(f"⏸️ {symbol}: Aguardando cooldown entre entradas (5 min).")
        return

    # =========================================
    # 3. BLOQUEIO POR LOSS STREAK
    # =========================================
    blocked, reason = utils.is_symbol_blocked(symbol)
    if blocked:
        return

    # =========================================
    # 4. LIMITE DIÁRIO DE TRADES POR ATIVO
    # =========================================
    if daily_trades_per_symbol[symbol] >= 4:
        return

    # =========================================
    # 5. VERIFICAÇÃO DE COTAÇÃO ATUAL
    # =========================================
    tick = mt5.symbol_info_tick(symbol)
    if not tick or tick.ask <= 0 or tick.bid <= 0:
        return

    # =========================================
    # 6. RECUPERAÇÃO DE INDICADORES
    # =========================================
    ind_data = bot_state.get_indicators(symbol)

    if not ind_data or ind_data.get("error"):
        return

    # =========================================
    # 7. LÓGICA DE PIRÂMIDE
    # =========================================
    existing_pos = mt5.positions_get(symbol=symbol)
    is_pyramiding = len(existing_pos) > 0
    pyr_count = 0

    if is_pyramiding:
        pos = existing_pos[0]
        pyr_count = pos.comment.count("PYR")

        # Verifica se a pirâmide é no mesmo sentido
        existing_side = "BUY" if pos.type == mt5.POSITION_TYPE_BUY else "SELL"
        if existing_side != side:
            logger.info(f"⏸️ {symbol}: Sinal oposto à posição atual. Ignorando pirâmide.")
            return

        atr = ind_data.get("atr", 0.10)
        current_price = tick.bid if existing_side == "BUY" else tick.ask
        profit_dist = abs(current_price - pos.price_open)

        # Lucro mínimo para pirâmide (precisa ter andado 1.5 ATR a favor)
        if profit_dist < (atr * 1.5):
            return

        # Limite máximo de níveis de pirâmide
        if pyr_count >= 3:
            return

    # =========================================
    # 8. CÁLCULO DE PREÇO E VOLUME
    # =========================================
    entry_price = tick.ask if side == "BUY" else tick.bid
    atr_val = ind_data.get("atr", 0.10)

    # Validação: ATR mínimo
    if atr_val < (entry_price * 0.003):
        atr_val = entry_price * 0.005

    stop_dist = atr_val * 2.0
    base_vol = utils.calculate_position_size_atr(symbol, entry_price, stop_dist)
    volume = base_vol * 0.5 if is_pyramiding else base_vol

    if volume <= 0:
        return

    # =========================================
    # 9. SL E TP DINÂMICOS
    # =========================================
    sl, tp = utils.calculate_dynamic_sl_tp(symbol, side, entry_price, ind_data)

    # =========================================
    # 10. FILTROS FINAIS
    # =========================================
    if not utils.validate_order_params(symbol, volume, entry_price, sl, tp):
        return
    if not utils.analyze_order_book_depth(symbol, side, volume):
        return
    if not utils.is_spread_acceptable(symbol):
        return

    # =========================================
    # 11. VALIDAÇÃO DE ORDEM
    # =========================================
    from validation import validate_and_create_order 
    order = validate_and_create_order(
        symbol=symbol, side=side, volume=volume, entry_price=entry_price, sl=sl, tp=tp
    )

    if not order:
        return False

    # =========================================
    # 12. EXECUÇÃO
    # =========================================
    daily_trades_per_symbol[symbol] += 1
    comment = f"XP3_PYR_{pyr_count + 1}" if is_pyramiding else "XP3_INIT"

    logger.info(
        f"🚀 ENVIANDO {'PIRÂMIDE' if is_pyramiding else 'ENTRADA'} {side} em {symbol} | "
        f"Vol: {volume:.0f} @ {entry_price:.2f}"
    )

    request = order.to_mt5_request(comment=comment)
    result = mt5.order_send(request)

    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        # ✅ ATUALIZA O COOLDOWN DE ENTRADA AQUI
        last_entry_time[symbol] = time.time()

        with position_open_times_lock:
            positions = mt5.positions_get(symbol=symbol)
            if positions:
                newest_pos = max(positions, key=lambda p: p.time)
                position_open_times[newest_pos.ticket] = time.time()

        with entry_indicators_lock:
            entry_indicators[symbol] = ind_data.copy()

        utils.send_telegram_trade(symbol, side, volume, entry_price, sl, tp, comment)
        
        # LOG NO TXT (Note que aqui exit_price=0 e pnl=0, ISSO É CORRETO PARA ABERTURA)
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
        return True
    else:
        logger.error(f"🚨 Falha ao enviar ordem {side} em {symbol}: {result.comment if result else 'Erro MT5'}")
        daily_trades_per_symbol[symbol] -= 1
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
def render_panel_enhanced():
    """
    Painel aprimorado com status de mercado em destaque
    """
    clear_screen()

    # ============================================
    # 1️⃣ OBTÉM STATUS DO MERCADO
    # ============================================
    market_status = get_market_status()

    # ============================================
    # 2️⃣ DADOS DO SISTEMA
    # ============================================
    current_indicators, top15_symbols = bot_state.snapshot

    with utils.mt5_lock:
        acc = mt5.account_info()

    if not acc:
        print("❌ Sem conexão com MT5")
        return

    now = datetime.now().strftime("%d/%m %H:%M:%S")
    pnl = acc.equity - acc.balance
    pnl_color = C_GREEN if pnl >= 0 else C_RED

    daily_pnl = acc.equity - equity_inicio_dia
    daily_pnl_pct = (
        (daily_pnl / equity_inicio_dia * 100) if equity_inicio_dia > 0 else 0
    )

    daily_target = config.PROFIT_TARGETS["daily"]["target_return"] * 100
    daily_progress = (
        min((daily_pnl_pct / daily_target) * 100, 100) if daily_target > 0 else 0
    )
    progress_color = (
        C_GREEN
        if daily_progress >= 100
        else C_YELLOW
        if daily_progress >= 50
        else C_RED
    )

    # ============================================
    # 3️⃣ CABEÇALHO COM STATUS EM DESTAQUE
    # ============================================
    print(f"{C_BOLD}╔{'═' * 96}╗{C_RESET}")
    print(f"║ {C_CYAN}🚀 XP3 PRO BOT - B3{C_RESET}  📅 {now}  ", end="")

    # Status de Circuit Breaker
    if trading_paused:
        print(f"{C_RED}🚨 PAUSADO (CB){C_RESET}", end="")
    else:
        print(
            f"{market_status['color']}{market_status['emoji']} {market_status['message']}{C_RESET}",
            end="",
        )

    # Preenche espaços restantes
    padding = 96 - len(
        f"🚀 XP3 PRO BOT - B3  📅 {now}  {market_status['emoji']} {market_status['message']}"
    )
    print(f"{' ' * padding}║")

    # ============================================
    # 4️⃣ LINHA DE STATUS DETALHADA
    # ============================================
    print(f"╠{'═' * 96}╣")

    # Monta mensagem de status
    if market_status["countdown"]:
        status_line = (
            f"{market_status['detail']} | Countdown: {market_status['countdown']}"
        )
    else:
        status_line = market_status["detail"]

    # Centraliza e adiciona padding
    status_padding = (96 - len(status_line)) // 2
    print(
        f"║{' ' * status_padding}{market_status['color']}{C_BOLD}{status_line}{C_RESET}",
        end="",
    )
    print(f"{' ' * (96 - len(status_line) - status_padding)}║")

    print(f"╠{'═' * 96}╣")

    # ============================================
    # 5️⃣ INFORMAÇÕES FINANCEIRAS
    # ============================================
    print(
        f"║ Equity: R$ {acc.equity:,.2f}  |  Balance: R$ {acc.balance:,.2f}  |  PnL: {pnl_color}{pnl:+,.2f}{C_RESET} {' ' * 30}║"
    )
    print(
        f"║ Posições: {mt5.positions_total()}/{config.MAX_SYMBOLS}  |  Risco/trade: {utils.get_current_risk_pct() * 100:.2f}% {' ' * 35}║"
    )
    print(
        f"║ Meta Diária: {progress_color}{daily_progress:.0f}%{C_RESET} ({daily_pnl_pct:+.2f}% / {daily_target:.1f}%) {' ' * 40}║"
    )
    print(f"╠{'═' * 96}╣")

    # ============================================
    # 6️⃣ LUCRO REALIZADO
    # ============================================
    lucro_no_bolso, total_fechadas = utils.calcular_lucro_realizado_txt()
    print(
        f"║ Lucro Realizado (Hoje): R$ {lucro_no_bolso:,.2f} ({total_fechadas} ordens) ║"
    )

    # ============================================
    # 7️⃣ STATUS DO SISTEMA
    # ============================================
    print("╠══════════ 🔧 STATUS DO SISTEMA ══════════╣")
    print(
        f"║ Correlação: {'✅ OK' if last_correlation_update else '⚠️ Nunca atualizada'} {' ' * 60}║"
    )
    print(
        f"║ Pesos adaptativos: {'✅ Carregados' if utils.symbol_weights else '⚠️ Vazios'} {' ' * 56}║"
    )
    print(f"║ Regime mercado: {detect_market_regime()} {' ' * 70}║")

    # Power Hour (só mostra se estiver no horário)
    power_hour_status = "🔥 ATIVA" if utils.is_power_hour() else "—"
    print(f"║ Power Hour: {power_hour_status} {' ' * 76}║")

    heat = get_portfolio_heat()
    heat_color = C_RED if heat > 0.7 else C_YELLOW if heat > 0.5 else C_GREEN
    print(f"║ Portfolio Heat: {heat_color}{heat:.3f}{C_RESET} {' ' * 70}║")

    # ML Status
    ml_trades = len(ml_optimizer.history) if hasattr(ml_optimizer, "history") else 0
    epsilon = getattr(ml_optimizer, "epsilon", 0.0)
    print(
        f"║ ML Trades: {ml_trades} | Epsilon: {epsilon:.3f} | Q-Table: {'Carregada' if os.path.exists('qtable.npy') else 'Nova'} {' ' * 40}║"
    )

    # ============================================
    # 8️⃣ ÚLTIMOS ALERTAS
    # ============================================
    print(f"╠{'═' * 96}╣")
    print(f"║ {C_YELLOW}🚨 ÚLTIMOS ALERTAS{C_RESET}{' ' * 76}║")

    with alerts_lock:
        recent = list(alerts)[:5]

    if not recent:
        print(f"║   {'(nenhum)':^92} ║")
    else:
        for _, msg in recent:
            msg_trunc = (msg[:90] + "...") if len(msg) > 90 else msg
            print(f"║   {msg_trunc:<92} ║")

    print(f"╠{'═' * 96}╣")

    # ============================================
    # 9️⃣ POSIÇÕES ABERTAS
    # ============================================
    print(
        f"║ {C_GREEN}💼 POSIÇÕES ABERTAS ({mt5.positions_total()}){C_RESET}{' ' * 68}║"
    )

    positions = mt5.positions_get() or []

    if not positions:
        print(f"║   {'(nenhuma posição aberta)':^92} ║")
    else:
        header = f"{'SYM':<6} {'DIR':<4} {'VOL':<8} {'ENTRY':<10} {'ATUAL':<10} {'P&L R$':<12} {'%':<7} {'STATUS':<15}"
        print(f"║ {header} ║")
        print(
            f"║ {'─' * 6} {'─' * 4} {'─' * 8} {'─' * 10} {'─' * 10} {'─' * 12} {'─' * 7} {'─' * 15} ║"
        )

        for p in positions:
            tick = mt5.symbol_info_tick(p.symbol)
            if not tick:
                continue

            side = "BUY" if p.type == mt5.POSITION_TYPE_BUY else "SELL"
            current_price = tick.bid if p.type == mt5.POSITION_TYPE_BUY else tick.ask

            if side == "BUY":
                pct = (current_price - p.price_open) / p.price_open * 100
            else:
                pct = (p.price_open - current_price) / p.price_open * 100

            # Status
            ind = current_indicators.get(p.symbol, {})
            atr = ind.get("atr", 0.01)
            if atr <= 0:
                atr = 0.01

            profit_atr = abs(current_price - p.price_open) / atr

            if profit_atr >= 2.5:
                status = "Trailing"
            elif profit_atr >= 1.0:
                status = "Breakeven"
            else:
                status = "Aguardando"

            line = (
                f"{p.symbol:<6} {side:<4} {p.volume:>7.0f} {p.price_open:>9.2f} "
                f"{current_price:>9.2f} {p.profit:>+11.2f} {pct:>+6.1f}% {status:<15}"
            )
            print(f"║ {line} ║")

    print(f"╠{'═' * 96}╣")

    # ============================================
    # 🔟 TOP 15 ELITE
    # ============================================
    print(f"║ {C_YELLOW}📊 TOP 15 ELITE{C_RESET}{' ' * 78}║")

    # Só mostra análise se estiver em horário de trading
    if market_status["trading_allowed"]:
        print(
            f"║ {'RK':<3} {'SYM':<6} {'SCORE':<6} {'DIR':<4} {'RSI':<5} {'ATR%':<6} {'HORA':<6} {'PREÇO':<8} {'CORR':<6} {'SETOR':<15} {'STATUS':<12} {'MOTIVO':<20} ║"
        )
        print(
            f"║ {'─' * 3} {'─' * 6} {'─' * 6} {'─' * 4} {'─' * 5} {'─' * 6} {'─' * 6} {'─' * 8} {'─' * 6} {'─' * 15} {'─' * 12} {'─' * 20} ║"
        )
    else:
        print(
            f"║ {'RK':<3} {'SYM':<6} {'PREÇO':<8} {'SETOR':<20} {'STATUS':<20} {' ' * 38}║"
        )
        print(f"║ {'─' * 3} {'─' * 6} {'─' * 8} {'─' * 20} {'─' * 20} {' ' * 38}║")

    indicators, top15_symbols = bot_state.snapshot

    if not top15_symbols:
        print(f"║   {'TOP15 vazio ou não carregado':^92} ║")
    else:
        positions_symbols = [p.symbol for p in mt5.positions_get() or []]
        current_sectors = get_sector_counts()

        for rank, sym in enumerate(top15_symbols, 1):
            ind = indicators.get(sym, {})
            if not ind:
                continue

            # ============================================
            # MODO 1: PRÉ-MERCADO (Versão Simplificada)
            # ============================================
            if not market_status["trading_allowed"]:
                df_last = utils.safe_copy_rates(sym, TIMEFRAME_BASE, 1)
                price = (
                    df_last["close"].iloc[-1]
                    if df_last is not None and not df_last.empty
                    else 0
                )
                price_str = f"{price:.2f}" if price > 0 else "-"

                sector_raw = config.SECTOR_MAP.get(sym, "UNKNOWN")
                sector = sector_raw[:19] + "…" if len(sector_raw) > 19 else sector_raw

                status_simple = (
                    "Aguardando abertura"
                    if market_status["status"] == "PRE_MARKET"
                    else "Monitorando"
                )

                line = f"{rank:<3} {sym:<6} {price_str:<8} {sector:<20} {C_DIM}{status_simple:<20}{C_RESET}"
                print(f"║ {line} {' ' * 38}║")

            # ============================================
            # MODO 2: MERCADO ABERTO (Versão Completa)
            # ============================================
            else:
                score = utils.calculate_signal_score(ind)
                long_signal = score >= config.MIN_SIGNAL_SCORE and ind.get(
                    "ema_fast", 0
                ) > ind.get("ema_slow", 0)
                short_signal = score >= config.MIN_SIGNAL_SCORE and ind.get(
                    "ema_fast", 0
                ) < ind.get("ema_slow", 0)
                dir_arrow = "↑" if long_signal else "↓" if short_signal else "-"
                side = "BUY" if long_signal else "SELL" if short_signal else None

                df_last = utils.safe_copy_rates(sym, TIMEFRAME_BASE, 1)
                last_time = (
                    df_last.index[-1].strftime("%H:%M")
                    if df_last is not None and not df_last.empty
                    else "N/A"
                )
                price_str = f"{ind.get('close', 0):.2f}" if ind.get("close") else "-"

                avg_corr = (
                    get_average_correlation_with_portfolio(sym, positions_symbols)
                    if config.ENABLE_CORRELATION_FILTER
                    else 0.0
                )
                corr_str = f"{avg_corr:.2f}" if avg_corr > 0 else "-"
                corr_color = (
                    C_RED if avg_corr > 0.7 else C_YELLOW if avg_corr > 0.5 else C_GREEN
                )

                sector_raw = config.SECTOR_MAP.get(sym, "UNKNOWN")
                sector = sector_raw[:14] + "…" if len(sector_raw) > 14 else sector_raw

                # Status
                if sym in positions_symbols:
                    status = "✔️ ABERTO"
                    status_color = C_GREEN
                    motive = "Posição ativa"
                elif not market_status["new_entries_allowed"]:
                    status = "⏸️ SEM ENTRADA"
                    status_color = C_YELLOW
                    motive = "Fora do horário"
                elif not additional_filters_ok(sym):
                    status = "⏸️ FILTRO"
                    status_color = C_RED
                    motive = "Vol/Gap/Spread"
                elif not macro_trend_ok(sym, side):
                    status = "⏸️ MACRO"
                    status_color = C_RED
                    motive = "Contra tendência"
                elif (
                    current_sectors.get(config.SECTOR_MAP.get(sym, "UNKNOWN"), 0)
                    >= config.MAX_PER_SECTOR
                ):
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

                motive_display = motive[:19] + "…" if len(motive) > 19 else motive

                log = ind.get("score_log", {})
                if log and status == "🟢 PRONTO":
                    parts = [f"{k}:{v}" for k, v in log.items() if abs(v) >= 10]
                    if parts:
                        enhanced_motive = " | ".join(parts[:3])
                        if len(parts) > 3:
                            enhanced_motive += " | ..."
                        motive_display = (
                            (enhanced_motive[:19] + "…")
                            if len(enhanced_motive) > 19
                            else enhanced_motive
                        )

                line = (
                    f"{rank:<3} {sym:<6} {score:>5.1f} {dir_arrow:<4} {ind.get('rsi', 0):>4.1f} "
                    f"{min(round(ind.get('atr_real', 0) * 5.3, 2), 9.99):>5.2f} {last_time:<6} {price_str:<8} "
                    f"{corr_color}{corr_str:<6}{C_RESET} {sector:<15} {status_color}{status:<12}{C_RESET} {motive_display:<20}"
                )
                print(f"║ {line} ║")

    print(f"╚{'═' * 96}╝")


# =========================
# 🔄 ATUALIZAR O FAST_LOOP (ADICIONAR PROFIT_LOCK)
# =========================
def fast_loop():
    """
    Loop principal com operação contínua
    """
    global trading_paused

    logger.info("⚙️ Fast Loop iniciado (modo contínuo)")
    logger.info("🔄 Inicializando pesos adaptativos e correlações...")
    utils.update_adaptive_weights()

    while True:  # ✅ Loop infinito (não depende de bot_should_run)
        try:
            health_monitor.heartbeat()

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
            # Adicione no fast_loop(), após build_portfolio_and_top15():
            save_top15_cache()
            save_system_status()
            time.sleep(5)

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


def force_test_trade(symbol, side="BUY"):
    """
    Ignora todos os filtros e tenta abrir uma ordem imediatamente para teste.
    """
    logger.info(f"🧪 INICIANDO TESTE FORÇADO: {side} em {symbol}")

    # 1. Obtém indicadores para o cálculo do SL/TP
    ind_data, _ = build_portfolio_and_top15()
    data = ind_data.get(symbol)

    if not data:
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
    """Salva TOP15 para o dashboard"""
    try:
        indicators, top15 = bot_state.snapshot

        data = []
        for rank, sym in enumerate(top15, 1):
            ind = indicators.get(sym, {})

            data.append(
                {
                    "Rank": rank,
                    "Símbolo": sym,
                    "Score": calculate_signal_score(ind),
                    "RSI": ind.get("rsi", 0),
                    "ATR%": ind.get("atr_real", 0) * 100,
                    "Preço": ind.get("close", 0),
                    "Setor": config.SECTOR_MAP.get(sym, "UNKNOWN"),
                    "Status": "✅ PRONTO"
                    if calculate_signal_score(ind) >= config.MIN_SIGNAL_SCORE
                    else "⏸️ AGUARDANDO",
                }
            )

        with open("top15_cache.json", "w") as f:
            json.dump(data, f)
    except Exception as e:
        logger.error(f"Erro ao salvar TOP15: {e}")

def update_bot_bridge():
    """Atualiza bot_bridge.json com máxima segurança contra dados corrompidos"""
    try:
        indicators, top15 = bot_state.snapshot
        
        # === LIMPEZA FORÇADA: Garante que todos os indicadores sejam dict ===
        safe_indicators = {}
        for sym, ind in indicators.items():
            if not isinstance(ind, dict):
                logger.warning(f"Indicador corrompido para {sym}: tipo {type(ind)}. Ignorando.")
                continue  # Pula símbolos com dados inválidos
            
            safe_ind = {}
            for k, v in ind.items():
                if isinstance(v, bool):
                    safe_ind[k] = int(v)
                elif isinstance(v, (np.integer, np.int64)):
                    safe_ind[k] = int(v)
                elif isinstance(v, (np.floating, np.float64)):
                    safe_ind[k] = float(v) if not np.isnan(v) else None
                elif isinstance(v, (int, float, str, type(None))):
                    safe_ind[k] = v
                else:
                    safe_ind[k] = str(v)  # fallback máximo
            safe_indicators[sym] = safe_ind
        
        if not top15:
            logger.warning("TOP15 vazio - não atualizando bridge")
            return
        
        data = {
            "timestamp": datetime.now().isoformat(),
            "top15": [],
            "indicators": safe_indicators
        }
        
        positions = mt5.positions_get() or []
        positions_symbols = {p.symbol for p in positions}
        
        for rank, sym in enumerate(top15, 1):
            ind = safe_indicators.get(sym, {})
            
            # Calcula score com fallback seguro
            try:
                score = utils.calculate_signal_score(ind)
            except:
                score = 0.0
            
            # Direção segura
            ema_fast = ind.get("ema_fast", 0)
            ema_slow = ind.get("ema_slow", 0)
            long_signal = score >= config.MIN_SIGNAL_SCORE and ema_fast > ema_slow
            short_signal = score >= config.MIN_SIGNAL_SCORE and ema_fast < ema_slow
            direction = "↑ LONG" if long_signal else "↓ SHORT" if short_signal else "—"
            
            # ATR% com fallback
            atr_real = ind.get("atr_real")
            atr_pct = round(atr_real * 5.3, 2) if isinstance(atr_real, (int, float)) and atr_real > 0 else 0.0
            atr_pct = min(atr_pct, 9.99)
            
            # Status
            if sym in positions_symbols:
                status = "✔️ ABERTO"
            elif score >= config.MIN_SIGNAL_SCORE:
                status = "🟢 PRONTO"
            else:
                status = "⏸️ AGUARDANDO"
            
            data["top15"].append({
                "rank": rank,
                "symbol": sym,
                "score": round(float(score), 1),
                "direction": direction,
                "rsi": round(float(ind.get("rsi", 0)), 1),
                "atr_pct": atr_pct,
                "price": round(float(ind.get("close", 0)), 2),
                "sector": config.SECTOR_MAP.get(sym, "UNKNOWN"),
                "status": status
            })
        
        # Escrita atômica
        temp_file = "bot_bridge.json.tmp"
        final_file = "bot_bridge.json"
        
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        # Substituição segura
        if os.path.exists(final_file):
            os.replace(temp_file, final_file)
        else:
            os.rename(temp_file, final_file)
            
        logger.info("✅ bot_bridge.json atualizado com sucesso (TOP15 sincronizado)")
        
    except Exception as e:
        logger.error(f"Erro crítico ao atualizar bot_bridge.json: {e}", exc_info=True)
        # Limpeza de emergência
        for tmp in ["bot_bridge.json.tmp", "bot_bridge.json"]:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except:
                    pass
                            
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


def telegram_polling_thread():
    bot = get_telegram_bot()
    if not bot:
        logger.error(
            "Não foi possível iniciar polling do Telegram - bot não inicializado"
        )
        return

    @bot.message_handler(commands=["lucro", "status", "resumo"])
    def handle_lucro(message):
        responder_comando_lucro(message)

    logger.info(
        "📡 Iniciando polling do Telegram... (comandos: /lucro, /status, /resumo)"
    )
    try:
        bot.infinity_polling(none_stop=True, interval=1, timeout=20)
    except Exception as e:
        logger.critical(f"Erro fatal no polling do Telegram: {e}")
        logger.info("Tentando reiniciar polling em 10 segundos...")
        time.sleep(10)
        telegram_polling_thread()  # tenta novamente


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

    # 1. Inicialização do MetaTrader 5
    if not mt5.initialize(path=config.MT5_TERMINAL_PATH):
        logger.critical(f"❌ Falha ao conectar no MT5: {config.MT5_TERMINAL_PATH}")
        return
    else:
        logger.info(f"✅ Conectado ao MT5 correto: {config.MT5_TERMINAL_PATH}")

    # 2. Carregamento de Persistência
    logger.info("📦 Carregando dados persistentes e otimizações...")
    utils.load_loss_streak_data()
    load_optimized_params()
    utils.load_adaptive_weights()

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

    try:
        while True:  # ✅ Loop infinito
            render_panel_enhanced()
            time.sleep(2)

    except KeyboardInterrupt:
        # ✅ Ctrl+C apenas para e salva, mas NÃO ENCERRA MT5
        print(f"\n{C_YELLOW}⏸️ Painel interrompido pelo usuário{C_RESET}")
        print(f"{C_YELLOW}ℹ️ O bot continua operando em background{C_RESET}")
        print(
            f"{C_YELLOW}Para encerrar completamente, feche a janela ou use Task Manager{C_RESET}"
        )

        # Salva dados
        utils.save_loss_streak_data()
        utils.save_adaptive_weights()

        # Mantém as threads rodando
        while True:
            time.sleep(3600)  # Aguarda indefinidamente


if __name__ == "__main__":
    main()
