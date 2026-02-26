#config.py
import os
import json
import yaml
from datetime import time
from pathlib import Path
from typing import Dict, Any
import logging

# ============================================
# LOGGING CONFIGURATION
# ============================================
# Ensure we capture errors to a dedicated file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("xp3_bot.log", encoding='utf-8', mode='a'),
        logging.FileHandler("errors.log", encoding='utf-8', mode='a', delay=True), # Dedicated error log
        logging.StreamHandler()
    ]
)

# ============================================
# ⚙️ CONFIGURAÇÕES GERAIS MT5
# ============================================
MAGIC_NUMBER = 987654
DEVIATION = 20
TIMEFRAME_BASE = "M5"
MT5_TERMINAL_PATH = r"C:\MetaTrader 5 Terminal\terminal64.exe"

# ===========================
# ✅ SISTEMA DE CONFIGURAÇÃO DINÂMICA VIA YAML
# ===========================

class ConfigManager:
    """
    Gerenciador de configurações com suporte a:
    - YAML editável via UI
    - Risk levels (Conservador/Agressivo)
    - Validação de parâmetros
    - Hot reload (sem reiniciar bot)
    """
    
    def __init__(self, config_file: str = "config.yaml"):
        self.config_file = Path(config_file)
        self.config: Dict[str, Any] = {}
        self.risk_level: str = "MODERADO"  # Padrão
        
        # Cria YAML se não existir
        if not self.config_file.exists():
            self._create_default_yaml()
        
        self.load()
    
    def _create_default_yaml(self):
        """Cria config.yaml com valores padrão"""
        default_config = {
            'mt5': {
                'terminal_path': r"C:\MetaTrader 5 Terminal\terminal64.exe"
            },
            
            'risk_levels': {
                'CONSERVADOR': {
                    'max_symbols': 5,
                    'risk_per_trade': 0.025,  # 0.2% (Kelly x0.2)
                    'max_daily_dd': 0.03,     # 3%
                    'min_win_rate': 0.65,     # 65%
                    'min_rr': 2.0
                },
                'MODERADO': {
                    'max_symbols': 8,
                    'risk_per_trade': 0.025,   # 0.5% (Kelly x0.2)
                    'max_daily_dd': 0.05,     # 5% (Limite CVM)
                    'min_win_rate': 0.60,     # 60%
                    'min_rr': 1.5
                },
                'AGRESSIVO': {
                    'max_symbols': 12,
                    'risk_per_trade': 0.025,   # 1% (Kelly x0.2)
                    'max_daily_dd': 0.07,     # 7%
                    'min_win_rate': 0.55,     # 55%
                    'min_rr': 1.2
                }
            },
            
            'ml': {
                'enabled': True,
                'min_confidence': 0.52,  # Reduced from 0.70 for realistic ensemble performance
                'model_type': 'ENSEMBLE',  # ENSEMBLE, LSTM, XGBOOST
                'retrain_frequency_days': 7,
                'min_samples_for_retrain': 500
            },
            
            'trading': {
                'enable_hedging': True,
                'enable_news_filter': True,
                'enable_correlation_filter': True
            }
        }
        
        with open(self.config_file, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False, indent=2)
        
        print(f"[OK] config.yaml criado em: {self.config_file.absolute()}")
    
    def load(self):
        """Carrega configurações do YAML"""
        try:
            with open(self.config_file, 'r') as f:
                self.config = yaml.safe_load(f)
            print(f"[OK] Configurações carregadas de {self.config_file}")
        except Exception as e:
            print(f"X Erro ao carregar config.yaml: {e}")
            self._create_default_yaml()
            self.load()
    
    def save(self):
        """Salva configurações no YAML"""
        try:
            with open(self.config_file, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False, indent=2)
            print(f"💾 Configurações salvas em {self.config_file}")
        except Exception as e:
            print(f"❌ Erro ao salvar config.yaml: {e}")
    
    def set_risk_level(self, level: str):
        """
        Define nível de risco: CONSERVADOR, MODERADO ou AGRESSIVO
        """
        if level not in ['CONSERVADOR', 'MODERADO', 'AGRESSIVO']:
            raise ValueError(f"Nível inválido: {level}")
        
        self.risk_level = level
        print(f"🎯 Nível de risco alterado para: {level}")
    
    def get(self, key_path: str, default=None):
        """
        Obtém valor com notação de ponto
        Ex: config.get('risk_levels.MODERADO.max_symbols')
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    @property
    def current_risk_params(self) -> Dict[str, Any]:
        """Retorna parâmetros do nível de risco atual"""
        return self.config['risk_levels'][self.risk_level]

    def update_dynamic_settings(self, win_rate: float):
        """
        ✅ Ajuste Dinâmico de Configurações (Runtime)
        Se WR > 60%, reduz RR mínimo para 1.3 (mais agressivo).
        Caso contrário, restaura o padrão do perfil de risco.
        """
        current_profile = self.config['risk_levels'][self.risk_level]
        
        # Pega o padrão original (fallback)
        default_rr = {
            'CONSERVADOR': 2.0,
            'MODERADO': 1.5,
            'AGRESSIVO': 1.2
        }.get(self.risk_level, 1.5)

        if win_rate >= 0.60:
            # Modo Performance: Permite RR menor
            if current_profile.get('min_rr') != 1.3:
                current_profile['min_rr'] = 1.3
                print(f"🚀 [Dynamic Config] WinRate {win_rate:.0%} > 60%: RR ajustado para 1.3")
        else:
            # Modo Normal: Restaura padrão se necessário
            if current_profile.get('min_rr') != default_rr:
                current_profile['min_rr'] = default_rr
                print(f"🛡️ [Dynamic Config] WinRate {win_rate:.0%} < 60%: RR restaurado para {default_rr}")

# ✅ INSTÂNCIA GLOBAL
config_manager = ConfigManager()

# ✅ BACKWARD COMPATIBILITY: Mantém variáveis antigas
MT5_TERMINAL_PATH = config_manager.get('mt5.terminal_path')
MAX_SYMBOLS = config_manager.current_risk_params['max_symbols']
MIN_RISK_PER_TRADE_PCT = 0.0025
MAX_RISK_PER_TRADE_PCT = 0.005
RISK_PER_TRADE_PCT = config_manager.current_risk_params['risk_per_trade']
try:
    RISK_PER_TRADE_PCT = float(RISK_PER_TRADE_PCT or MIN_RISK_PER_TRADE_PCT)
except:
    RISK_PER_TRADE_PCT = MIN_RISK_PER_TRADE_PCT
RISK_PER_TRADE_PCT = max(MIN_RISK_PER_TRADE_PCT, min(RISK_PER_TRADE_PCT, MAX_RISK_PER_TRADE_PCT))
MAX_DAILY_DRAWDOWN_PCT = config_manager.current_risk_params['max_daily_dd']
MAX_PER_SECTOR = 2
ELITE_SYMBOLS_JSON_PATH = config_manager.get('optimizer.elite_symbols_json_path', 'optimizer_output/elite_symbols_latest.json')
# ✅ NOVOS PARÂMETROS ML
ENABLE_ML_SIGNALS = config_manager.get('ml.enabled', True)
ML_MIN_CONFIDENCE = config_manager.get('ml.min_confidence', 0.52)  # Reduced from 0.78 to 0.65 for realistic ensemble
ML_MODEL_TYPE = config_manager.get('ml.model_type', 'ENSEMBLE')
ML_MIN_SAMPLES_FOR_RETRAIN = config_manager.get('ml.min_samples_for_retrain', 500)
ML_RETRAIN_THRESHOLD = 100  # Retreino após 100 trades (mais estável)
ML_Q_STATES = 5000  # Aumentado para 5000 estados
ML_TRAIN_PER_SYMBOL = config_manager.get('ml.train_per_symbol', True)
ML_PER_SYMBOL_MIN_SAMPLES = config_manager.get('ml.per_symbol_min_samples', 50)
WR_RESET_GRACE_TRADES = config_manager.get('risk.winrate_reset_grace_trades', 5)

# ===========================
# 💰 PARÂID METROS DE RISCO (FUTUROS)
# ===========================
RISK_PER_TRADE_PCT = 0.006              # 0.6% por trade (futuros)
MIN_RR = 2.5                             # R:R mínimo para futuros
MAX_ATR_PCT = 6.0                        # ATR máximo permitido (%)
PYRAMID_MAX_LEGS = 3                     # Máximo de pernas em pirâmide
FUTURES_RISK_MULTIPLIER = 1.5            # Multiplicador de risco para futuros

# ✅ KELLY CRITERION & POSITION SIZING
KELLY_MULTIPLIER = 0.3  # Fractional Kelly (0.3x) for capital preservation
KELLY_MIN_TRADES_FOR_CALC = 30  # Minimum trades to calculate dynamic Kelly

# ✅ ATR VOLATILITY FILTERS
MAX_ATR_PCT = 5.0  # Base ATR limit (5.0% for blue chips)
ADAPTIVE_ATR_FILTER = True  # Enable adaptive ATR based on ADX
MAX_ATR_PCT_HIGH_ADX = 7.0  # Allow up to 7% ATR when ADX > 40 (strong trend)
MIN_VOLATILITY_TICKS = 12  # Mínimo de ticks de ATR para operar (evita custos > lucro)


# ===========================
# 🛡️ CONTROLES COMERCIAIS
# ===========================
MAX_TOTAL_EXPOSURE = int(config_manager.get('risk.max_total_exposure', 1_000_000))
MAX_SPREAD_TICKS = 4                   # Spread máximo permitido em ticks
DAILY_PROFIT_TARGET_PCT = 0.02          # Meta diária de lucro (2%)
BLOCK_AFTER_CONSECUTIVE_LOSSES = 2      # Parar após N perdas consecutivas
MARKET_HOURS_BUFFER_OPEN = 0           # Minutos após abertura para iniciar
MARKET_HOURS_BUFFER_CLOSE = 0          # Minutos antes do fechamento para parar
SLIPPAGE_ALERT_TICKS = 3                # Alertar se slippage > N ticks
MAX_VOL_THRESHOLD = 2.0
DAILY_VOLUME_LIMIT = 1000000000  # R$ 1 bilhão (limite financeiro diário)

# HORÁRIOS ESPECÍFICOS (FUTUROS)
# ===========================
HORARIOS_OPERACAO = {
    "FUTUROS": [(time(9, 0), time(18, 0))],           # Horário completo de futuros
    "SO_FUTUROS": [(time(9, 0), time(10, 0)), (time(17, 0), time(18, 0))]  # Apenas janelas específicas
}


# ===========================
# CONFIGURAÇÕES DE FUTUROS
# ===========================
# Mapeamentos de subsetor removidos (específicos de ações)


# ============================================
# 🎯 APENAS FUTUROS - NÃO ADICIONAR AÇÕES AQUI
# ============================================
# Mapa de setores - APENAS ÍNDICES FUTUROS B3
SECTOR_MAP = {
   "WIN$N": "FUTUROS",
    "IND$N": "FUTUROS",
    "WDO$N": "FUTUROS",
    "DOL$N": "FUTUROS",
    "WSP$N": "FUTUROS",
    "CCM$N": "FUTUROS",
    "BGI$N": "FUTUROS",
    "ICF$N": "FUTUROS",
    "DI1$N": "FUTUROS",
    "BIT$N": "FUTUROS",
    "T10$N": "FUTUROS",
}


ACTIVE_FUTURES = {}


# Lista de símbolos proxy (usada em alguns módulos antigos - pode manter)
PROXY_SYMBOLS = [
    
]

SCORE_WEIGHTS = {
    "EMA": 1.0,
    "RSI_ADX": 1.0,
    "VWAP": 1.0,
    "MACRO": 1.0,
    "ATR": 1.0,
    "CORR": 1.0,
}

MIN_SIGNAL_SCORE = 35

# ===========================
# HORÁRIOS DE OPERAÇÃO
# ===========================
TRADING_START = "09:25"  # Após estabilização da abertura
NO_ENTRY_AFTER = "17:15"  # Fim das entradas (antes do fechamento nervoso)
CLOSE_ALL_BY = "17:40"  # FECHAMENTO FORÇADO (nunca posar no after)
NO_ENTRY_BEFORE_CLOSE_MINUTES = 15  # Bloqueia novas entradas quando faltar pouco p/ fechar
DAILY_RESET_TIME = "00:00"  # Reset diário do circuit breaker
DAY_ONLY_MODE = True
FRIDAY_NO_ENTRY_AFTER = "15:30"
FRIDAY_CLOSE_ALL_BY = "15:30"
FRIDAY_NO_ENTRY_BEFORE_CLOSE_MINUTES = 15
FRIDAY_MARKET_HOURS_BUFFER_CLOSE = 0
FRIDAY_RISK_REDUCE_AFTER = "16:30"
FRIDAY_RISK_FACTOR_MULT = 0.60
FRIDAY_DISABLE_PYRAMID_AFTER = "16:30"
# ✅ Pausa operacional entre 11:45 e 13:15
TRADING_LUNCH_BREAK_START = "11:45"
TRADING_LUNCH_BREAK_END = "13:15"
FUTURES_CLOSE_ALL_BY = "17:50"
FUTURES_AFTERMARKET_START = "16:00"
FUTURES_AFTERMARKET_END = "17:50"
MAX_ACCEPTABLE_GAP_PCT = 0.015
# ===========================
# GESTÃO DE RISCO
# ===========================
ENABLE_NEWS_FILTER = True
NEWS_BLOCK_BEFORE_MIN = 30          # Bloqueia 30min antes do evento
NEWS_BLOCK_MEDIUM_TOO = False       # True = bloqueia também eventos Medium (ex: IPCA)
NEWS_INCLUDE_MEDIUM = True
ENABLE_NEWS_FALLBACK_WINDOWS = True
NEWS_FALLBACK_BLACKOUT_WINDOWS = [
    {"start": "08:55", "end": "10:20", "label": "Abertura (volatilidade e leilão)"},
    {"start": "14:25", "end": "14:40", "label": "Macro EUA (horário 14:30)"},
    {"start": "15:55", "end": "18:10", "label": "Macro BR (horário 16:00)"},
]
ENABLE_NEWS_SENTIMENT_BLOCK = False
NEWS_SENTIMENT_NEG_THRESHOLD = -0.70
NEWS_SENTIMENT_BLOCK_MINUTES = 60

ML_MODE = config_manager.get('ml.mode', 'advisory')  # advisory|gate
ML_ADVISORY_HARD_BLOCK = config_manager.get('ml.advisory_hard_block', 0.82)
ML_ADVISORY_SOFT_RISK = config_manager.get('ml.advisory_soft_risk', 0.70)
ML_ADVISORY_SOFT_RISK_FACTOR = config_manager.get('ml.advisory_soft_risk_factor', 0.60)
DEFAULT_TIMEFRAME = "M5"

ENTRY_SCORE_DELTA_MORNING = config_manager.get('entry.score_delta_morning', 5)
ENTRY_SCORE_DELTA_LUNCH = config_manager.get('entry.score_delta_lunch', 10)
ENTRY_SCORE_DELTA_AFTERNOON = config_manager.get('entry.score_delta_afternoon', 0)
MAX_RISK_PER_SYMBOL_PCT = 0.02  # Máximo 2% da equity por papel
MAX_CAPITAL_USAGE_PCT = 0.35    # Máximo 35% do equity total em margem usada
MAX_SECTOR_EXPOSURE = 0.30  # Máx 30% do capital em 1 setor
MAX_SECTOR_EXPOSURE_PCT = 0.25  # Máx 30% do capital em 1 setor
SYMBOL_BLOCK_LOSS_PCT = 0.025  # Bloqueia ativo após perda de 2.5%
SYMBOL_BLOCK_HOURS = 72
SYMBOL_MAX_CONSECUTIVE_LOSSES = 2  # Bloqueia ativo após 3 perdas consecutivas
SYMBOL_COOLDOWN_HOURS = 24
# ===========================
# FUTUROS B3 (WIN/WDO)
# ===========================
FUTURES_MAX_CONTRACTS = 10
WIN_POINT_VALUE = 0.20
WDO_POINT_VALUE = 10.0
FUTURE_FEE_PER_CONTRACT = 1.0
# NOVOS: Hedging e Risco
HEDGE_UNWIND_DD_THRESHOLD = 0.03    # Desfazer hedge se DD < 3%
HEDGE_UNWIND_VIX_THRESHOLD = 25     # Desfazer hedge se VIX < 25
VIX_THRESHOLD_RISK_OFF = 30         # Aciona modo defensivo se VIX > 30
VIX_THRESHOLD_PROTECTION = 35       # Aciona modo proteção se VIX > 35
MIN_RR_HIGH_VOL = 2.5               # R:R mínimo em alta volatilidade
MAX_PORTFOLIO_IBOV_CORR = 0.85      # Rejeita se correlação com IBOV > 0.85
MAX_DAILY_DD_STOP = 0.05            # Auto-stop se DD diário > 5%
MAX_PORTFOLIO_HEAT = 0.65           # Bloqueia novas entradas se 'heat' > 0.65
MIN_FINANCIAL_VOLUME_R = 20_000_000 # R$ mínimo de volume financeiro médio (M15)
DAILY_STOP_MONEY = 2000              # Stop diário absoluto em R$
ML_MIN_TRADES_ENABLE = 200           # ML desativado até 200 trades

# NOVOS: A/B Testing e Infra
AB_TEST_ENABLED = True
AB_TEST_GROUPS = {
    "A": {"min_confidence": 0.61, "signal_threshold": 61},
    "B": {"min_confidence": 0.65, "signal_threshold": 65}
}
REDIS_CACHE_TTL_TICK = 10    # TTL em segundos para ticks
REDIS_CACHE_TTL_INFO = 10    # TTL em segundos para symbol_info/indicadores

# =========================================================
# 🌐 POLYGON.IO API (DADOS B3)
# =========================================================
POLYGON_API_KEY = "xrE09LEWJYBZfQcV57pCvsw4aqkOiqbz"
POLYGON_BASE_URL = "https://api.polygon.io"
POLYGON_CACHE_TTL = 30  # Cache de 30 segundos para dados Polygon

NEWS_API_KEY = "c39162901d1a45eeaad80d3c3f6f8c1e"  # NewsAPI.org


# Slippage para futuros B3 (em ticks)
SLIPPAGE_TICKS = {
    "WIN$N": 3,      # Mini Índice: ~3 ticks
    "WDO$N": 2,      # Mini Dólar: ~2 ticks

    "DEFAULT": 3     # Padrão para outros futuros
}
MAX_SPREAD_FUTURE_POINTS = 20           # Máx spread em pontos para futuros
FUTURE_FEE_PER_CONTRACT = 1.0           # Taxa por contrato

# =========================================================
# 📊 PARÂMETROS DINÂMICOS POR WIN RATE
# =========================================================
# Ajusta agressividade baseado no win rate atual

WIN_RATE_TIERS = {
    # Win Rate >= 70%: Modo agressivo
    "HIGH": {
        "min_win_rate": 0.70,
        "kelly_multiplier": 0.4,
        "max_symbols": 12,
        "min_rr": 1.2,
        "max_daily_dd": 0.04,         # DD maior permitido
        "signal_threshold": 58        # Threshold menor
    },
    # Win Rate 60-70%: Modo normal
    "MEDIUM": {
        "min_win_rate": 0.60,
        "kelly_multiplier": 0.12,
        "max_symbols": 8,
        "min_rr": 1.5,
        "max_daily_dd": 0.03,
        "signal_threshold": 61
    },
    # Win Rate 50-60%: Modo conservador
    "LOW": {
        "min_win_rate": 0.50,
        "kelly_multiplier": 0.10,
        "max_symbols": 5,
        "min_rr": 2.0,
        "max_daily_dd": 0.02,
        "signal_threshold": 65
    },
    # Win Rate < 50%: Modo proteção
    "CRITICAL": {
        "min_win_rate": 0.0,
        "kelly_multiplier": 0.10,
        "max_symbols": 3,
        "min_rr": 2.5,
        "max_daily_dd": 0.01,
        "signal_threshold": 70
    }
}

# =========================================================
# 🎯 MODOS DE OPERAÇÃO
# =========================================================

OPERATION_MODES = {
    "NORMAL": {
        "description": "Operação padrão",
        "allow_new_entries": True,
        "allow_pyramiding": False,
        "max_concurrent_positions": 6,
        "profit_lock_pct": 0.012
    },
    "AGGRESSIVE": {
        "description": "Modo agressivo - alta confiança",
        "allow_new_entries": True,
        "allow_pyramiding": True,
        "max_concurrent_positions": 12,
        "profit_lock_pct": 0.015
    },
    "DEFENSIVE": {
        "description": "Modo defensivo - mercado volátil",
        "allow_new_entries": True,
        "allow_pyramiding": False,
        "max_concurrent_positions": 5,
        "profit_lock_pct": 0.008
    },
    "PROTECTION": {
        "description": "Proteção de Capital (VIX > 35 ou DD > 4%)",
        "allow_new_entries": False,
        "allow_pyramiding": False,
        "max_concurrent_positions": 0,
        "profit_lock_pct": 0.005
    },
    # ✅ NOVO MODO: TEST
    "TEST": {
        "description": "Modo de Teste (Simulação Controlada)",
        "allow_new_entries": True,
        "allow_pyramiding": False,
        "max_concurrent_positions": 100,
        "profit_lock_pct": 0.01,
        "risk_free": True # Flag customizada para bots de teste
    }
}

# =========================
# 🔌 INTEGRAÇÃO POLYGON.IO (Agora Massive.com)
# =========================
# Nota: api.polygon.io continua funcionando (alias para api.massive.com)
POLYGON_BASE_URL = "https://api.polygon.io"
POLYGON_CACHE_TTL = 60 # Cache de 1 minuto

# Modo atual (pode ser alterado dinamicamente)
CURRENT_OPERATION_MODE = "NORMAL"


def get_params_for_win_rate(win_rate: float) -> dict:
    """
    Retorna parâmetros dinâmicos baseados no win rate atual.
    """
    if win_rate >= 0.70:
        return WIN_RATE_TIERS["HIGH"]
    elif win_rate >= 0.60:
        return WIN_RATE_TIERS["MEDIUM"]
    elif win_rate >= 0.50:
        return WIN_RATE_TIERS["LOW"]
    else:
        return WIN_RATE_TIERS["CRITICAL"]


def get_current_mode_params() -> dict:
    """
    Retorna parâmetros do modo de operação atual.
    """
    return OPERATION_MODES.get(CURRENT_OPERATION_MODE, OPERATION_MODES["NORMAL"])


def set_operation_mode(mode: str):
    """
    Altera o modo de operação.
    """
    global CURRENT_OPERATION_MODE
    if mode in OPERATION_MODES:
        CURRENT_OPERATION_MODE = mode
        return True
    return False


MAX_TRADE_DURATION_CANDLES = 40  # Time-stop
# config.py - ADICIONAR
ADAPTIVE_THRESHOLDS = {
    "RISK_ON": {
        "min_signal_score": 45,
        "min_adx": 12,
        "min_volume_ratio": 0.40,
    },
    "RISK_OFF": {
        "min_signal_score": 50,
        "min_adx": 15,
        "min_volume_ratio": 0.50,
        "anti_chop_cooldown": 240,
    }
}

# ===========================
# PYRAMIDING
# ===========================
ENABLE_PYRAMID = True
PYRAMID_MAX_LEGS = 3
PYRAMID_ATR_DISTANCE = 1.0  # Segunda perna só após +1.0 ATR a favor
PYRAMID_MIN_PCT_DISTANCE = 0.0015
PYRAMID_MINUTES_BETWEEN_ADDS = 10
PYRAMID_REQUIRE_PROFIT = True
PYRAMID_RISK_SPLIT = [0.6, 0.4]
PYRAMID_REQUIREMENTS = {
    "min_adx": 28,  # ADX > 30 (tendência forte confirmada)
    "max_rsi_long": 65,  # RSI não sobrecomprado (compra)
    "min_rsi_short": 35,  # RSI não sobrevendido (venda)
    "volume_ratio": 1.3,  # Volume 20% acima da média
    "time_since_entry": 45,  # Mínimo 45 min desde primeira perna
}

# ===========================
# STOP LOSS / TAKE PROFIT
# ===========================
SL_ATR_MULTIPLIER = 0.8  # SL inicial = preço ± ATR × 2.0
TP_ATR_MULT = 3.0  # TP opcional (não usado atualmente, mas disponível)
TRAILING_STEP_ATR_MULTIPLIER = 1.0

# ===========================
# FILTROS PROFISSIONAIS B3
# ===========================
MIN_AVG_VOLUME_20 = 20000  # Volume médio 20 períodos mínimo
MIN_LIQUIDITY_THRESHOLD = 7e5 # Liquidez mínima de 1M
MAX_GAP_OPEN_PCT = 0.03  # Gap de abertura > 3% → bloqueia entrada
MIN_AVG_VOLUME = 4000
VOLATILITY_MIN_MULT = 0.60
VOLATILITY_MAX_MULT = 2.50
RSI_OVERSOLD = 28
RSI_OVERBOUGHT = 72
RSI_EXTREME_OVERSOLD = 22
RSI_EXTREME_OVERBOUGHT = 78
RSI_EXHAUSTION_DEFAULT = 70
RSI_EXHAUSTION_INDEX = 80
RSI_EXHAUSTION_HIGH_SCORE_LIMIT = 75
RSI_EXHAUSTION_HIGH_SCORE_MIN_SCORE = 80
RSI_EXHAUSTION_DEFAULT_SELL = 30
RSI_EXHAUSTION_INDEX_SELL = 20
RSI_EXHAUSTION_HIGH_SCORE_LIMIT_SELL = 25
MIN_BUY_AGGRESSION_BALANCE = 0.08
MIN_SELL_AGGRESSION_BALANCE = -0.08

# ===========================
# FILTRO DE CORRELAÇÃO
# ===========================
ENABLE_CORRELATION_FILTER = True
MIN_CORRELATION_SCORE_TO_BLOCK = 0.70
CORRELATION_LOOKBACK_DAYS = 60

# ===========================
# FILTRO MACRO
# ===========================
MACRO_TIMEFRAME = "H1"
MACRO_EMA_LONG = 200
EMA200_SLOPE_MIN = 0.05
DIST_EMA200_ATR_THRESH = 2.0
EXIT_AT_EMA200 = False

# ===========================
# MODOS E CONTROLES
# ===========================
TRADE_BOTH_DIRECTIONS = True
FAST_LOOP_INTERVAL_SECONDS = 1.0
CORR_UPDATE_INTERVAL = 1800  # 30 minutos

# ===========================
# OTIMIZADOR
# ===========================
WFO_OOS_RATIO = 0.30
ENABLE_MONTE_CARLO = True

OPTIMIZER_OUTPUT = "optimizer_output"
OPTIMIZER_HISTORY_FILE = os.path.join(OPTIMIZER_OUTPUT, "history.json")
# ===========================
# GESTÃO AVANÇADA DE SAÍDA
# ===========================
ENABLE_BREAKEVEN = True
BREAKEVEN_ATR_MULT = 1.0  # Move SL para entrada após +1.0 ATR

ENABLE_PARTIAL_CLOSE = True
PARTIAL_CLOSE_ATR_MULT = 2.0  # Fecha 50% da posição em +2.0 ATR
PARTIAL_PERCENT = 0.5  # % da posição a fechar
MAX_TRADE_DURATION_CANDLES = 80
ENABLE_TRAILING_STOP = True
TRAILING_ATR_MULT_INITIAL = 2.0
TRAILING_ATR_MULT_TIGHT = 1.3
VWAP_OVEREXT_STD_MULT = 2.0
SPREAD_LOOKBACK_BARS = 10
B3_FEES_PCT = 0.0003
AVG_SPREAD_PCT_DEFAULT = 0.001
MIN_ADX_SLOPE = 0.5
MAX_SPREAD_TREND_PCT = 0.03
MAX_SPREAD_PCT = 0.15
PER_ASSET_THRESHOLDS = {
    "WIN": {
        "MIN_ADX_SLOPE": 0.6,
        "MAX_SPREAD_TREND_PCT": 0.03,
        "MAX_SPREAD_PCT": 0.15
    },
    "IND": {
        "MIN_ADX_SLOPE": 0.5,
        "MAX_SPREAD_TREND_PCT": 0.03,
        "MAX_SPREAD_PCT": 0.12
    },
    "WDO": {
        "MIN_ADX_SLOPE": 0.4,
        "MAX_SPREAD_TREND_PCT": 0.02,
        "MAX_SPREAD_PCT": 0.10
    },
    "DOL": {
        "MIN_ADX_SLOPE": 0.4,
        "MAX_SPREAD_TREND_PCT": 0.02,
        "MAX_SPREAD_PCT": 0.10
    }
}
# ===========================
# NOTIFICAÇÕES TELEGRAM
# ===========================
ENABLE_TELEGRAM_NOTIF = True
TELEGRAM_BOT_TOKEN = (
    "8474186435:AAGpRE6ou0a-aUqKATKRI4mVpzxYDotWeuQ"
)
TELEGRAM_CHAT_ID = 8400631213
ENABLE_TELEGRAM_REJECTION_SUMMARY = False
EOD_REPORT_ENABLED = True
EOD_REPORT_TIME = "16:55"  # Seu chat_id (número inteiro)
# ===========================

# =========================nn
# ⏰ TIME-AWARE SCORING
# =========================

TIME_SCORE_RULES = {
    "OPEN": {
        "start": "10:30",
        "end": "11:30",
        "adx_min": 18,
        "min_score": 40,
        "atr_max": 8.0,
        "min_volume_ratio": 1.1,     # Volume atual > 130% da média de 20 períodos
        "require_vwap_proximity": True,  # Preço perto do VWAP intraday (±1%)
        "min_momentum": 0.0007,  # Momentum mínimo mais exigente
    },
    "MID": {
        "start": "11:30",
        "end": "14:30",
        "adx_min": 18,
        "min_score": 35,
        "atr_max": 10.0,
        "min_volume_ratio": 1.05
    },
    "LATE": {
        "start": "14:30",
        "end": "16:55",
        "adx_min": 18,
        "min_score": 35,
        "atr_max": 12.0,
    },
}

ADAPTIVE_FILTERS = {
    "spread": {
        "normal": 0.25,  # 10:00-15:30 (era 0.10)
        "power_hour": 0.35,  # 15:30-18:00 (era 0.12)
    },
    "book_depth": {
        "normal": 0.50,  # Exige 50% do volume
        "power_hour": 0.20,  # Exige apenas 20%
    },
    "volume_impact": {
        "normal": 0.20,  # Máx 20% do volume médio
        "power_hour": 0.35,  # Máx 35% (maior tolerância)
    },
}

# =========================
# ⚡ POWER-HOUR MODE
# =========================

POWER_HOUR = {
    "enabled": True,
    "start": "15:30",
    "end": "16:55",
    "min_atr_pct": 0.8,
    "min_volume_ratio": 0.45,
    "score_boost": 10,
}

ORDER_BOOK_DEPTH_MULTIPLIER = 3
ORDER_BOOK_DEPTH_MULTIPLIER_POWER_HOUR = 3
LUNCH_MIN_VOLUME_RATIO = 0.40
MACRO_OVERRIDE_ADX = 30
MACRO_OVERRIDE_RISK_FACTOR = 0.70

# =========================
# 🚀 VOLATILITY BREAKOUT
# =========================

VOL_BREAKOUT = {
    "enabled": True,
    "lookback": 20,
    "atr_expansion": 1.20,
    "volume_ratio": 0.55,
    "score_boost": 15,
}

# ===========================
# 🎯 TARGETS DINÂMICOS POR REGIME
# ===========================

TP_RULES = {
    "TRENDING": {
        "min_adx": 30,
        "tp_mult": 4.5,  # 1:2.25 R:R
        "partial_mult": 2.5,
        "trailing_initial": 3.0,
        "trailing_tight": 1.8,
    },
    "RANGING": {
        "min_adx": 0,
        "max_adx": 25,
        "tp_mult": 2.0,  # 1:1.25 R:R (conservador)
        "partial_mult": 1.8,
        "trailing_initial": 2.0,
        "trailing_tight": 1.2,
    },
    "BREAKOUT": {
        "vol_expansion": 1.3,  # ATR 30% acima da média
        "tp_mult": 5.0,  # 1:2.5 R:R (agressivo)
        "partial_mult": 3.0,
        "trailing_initial": 3.5,
        "trailing_tight": 2.0,
    },
}

# ===========================
# 📈 METAS DE PROFIT FACTOR
# ===========================

PROFIT_TARGETS = {
    "daily": {
        "min_pf": 1.5,  # Profit Factor mínimo do dia
        "target_return": 0.015,  # 1.5% ao dia
        "max_return": 0.03,  # 3% ao dia (conservadorismo)
    },
    "weekly": {
        "target_return": 0.06,  # 6% na semana
        "max_dd": 0.04,  # 4% de drawdown máximo
    },
    "monthly": {
        "target_return": 0.10,  # 20% ao mês (agressivo mas possível)
        "min_sharpe": 1.5,  # Sharpe Ratio > 1.5
    },
}

# ===========================
# 🛡️ PROFIT PROTECTION
# ===========================

PROFIT_LOCK = {
    "enabled": True,
    "daily_target_pct": 0.01,  # 2% de lucro no dia
    "lock_pct": 0.70,  # Trava 70% do lucro
    "reduce_risk": True,  # Reduz risco para 0.5% após meta
    "close_winners_only": True,
    "tighten_trailing": True,
    "tighten_trailing_atr_mult": 1.0,
    "min_minutes_between_actions": 5,
}

# ===========================
# PARÂMETROS OTIMIZADOS MANUAIS (ELITE)
# ===========================
ELITE_SYMBOLS = {
    # ===========================
    # ÍNDICES FUTUROS B3
    # ===========================
    
     "WIN$N": {
        "parameters": {
            "ema_short": 6,
            "ema_long": 54,
            "rsi_low": 40,
            "rsi_high": 67,
            "adx_threshold": 20,
            "sl_atr_multiplier": 1.6,
            "tp_ratio": 1.0
        },
        "performance_targets": {
            "expectancy_points": 38.5,
            "profit_factor": 192.87,
            "sharpe_ratio": 3.42,
            "win_rate_target": 21.4
        },
        "safety_thresholds": {
            "max_drawdown_allowed": 2.9,
            "stop_trading_at_loss": -4.0,
            "min_trades_for_validation": 30
        }
    },
    "WDO$N": {
        "parameters": {
            "ema_short": 9,
            "ema_long": 21,
            "rsi_low": 32,
            "rsi_high": 55,
            "adx_threshold": 18,
            "sl_atr_multiplier": 2.2,
            "tp_ratio": 2.5
        },
        "performance_targets": {
            "expectancy_points": 4.2,
            "profit_factor": 1.72,
            "sharpe_ratio": 1.56,
            "win_rate_target": 15.0
        },
        "safety_thresholds": {
            "max_drawdown_allowed": 9.3,
            "stop_trading_at_loss": -12.0,
            "min_trades_for_validation": 25
        }
    },
    "IND$N": {
        "parameters": {
            "ema_short": 6,
            "ema_long": 54,
            "rsi_low": 40,
            "rsi_high": 67,
            "adx_threshold": 20,
            "sl_atr_multiplier": 2.8,
            "tp_ratio": 1.5
        },
        "performance_targets": {
            "expectancy_points": 150.0,
            "profit_factor": 17.92,
            "sharpe_ratio": 3.00,
            "win_rate_target": 11.1
        },
        "safety_thresholds": {
            "max_drawdown_allowed": 15.0,
            "stop_trading_at_loss": -20.0,
            "min_trades_for_validation": 20
        }
    },
    "DOL$N": {
        "parameters": {
            "ema_short": 11,
            "ema_long": 30,
            "rsi_low": 27,
            "rsi_high": 69,
            "adx_threshold": 15,
            "sl_atr_multiplier": 1.5,
            "tp_ratio": 2.2
        },
        "performance_targets": {
            "expectancy_points": 5.5,
            "profit_factor": 3.06,
            "sharpe_ratio": 1.87,
            "win_rate_target": 12.0
        },
        "safety_thresholds": {
            "max_drawdown_allowed": 8.3,
            "stop_trading_at_loss": -10.0,
            "min_trades_for_validation": 30
        }
    },
    "WSP$N": {
        "parameters": {
            "ema_short": 6,
            "ema_long": 54,
            "rsi_low": 40,
            "rsi_high": 67,
            "adx_threshold": 20,
            "sl_atr_multiplier": 1.6,
            "tp_ratio": 1.0
        },
        "performance_targets": {
            "expectancy_points": 8.0,
            "profit_factor": 52.61,
            "sharpe_ratio": 2.77,
            "win_rate_target": 14.3
        },
        "safety_thresholds": {
            "max_drawdown_allowed": 1.0,
            "stop_trading_at_loss": -2.5,
            "min_trades_for_validation": 20
        }
    }

}

# Listas de blue chips removidas (específicas de ações)

ELITE_ASSETS = {}

def get_elite_settings(path: str = "elite_params.json") -> dict:
    global ELITE_ASSETS
    try:
        with open(path, "r") as f:
            data = json.load(f)
        if isinstance(data, dict):
            ELITE_ASSETS = data
            return data
    except Exception:
        pass
    return ELITE_ASSETS or {}

LOW_LIQUIDITY_SYMBOLS = {}
LIQUIDITY_THRESHOLD_PCT = 0.5 
# ============================================
# 🔥 PRIORIDADE 1 - ANTI-CHOP
# ============================================

# Cooldown após SL (evita reentrar em movimento lateral)
ANTI_CHOP = {
    "enabled": True,
    "cooldown_after_sl_minutes": 180,  # 30 min após SL antes de reentrar
    "min_range_pct": 1.2,  # Preço precisa andar ≥0.8% antes de nova entrada
    "max_consecutive_losses": 2,  # Máx 2 perdas seguidas → bloqueia ativo
    "block_duration_hours": 6,  # Bloqueia 6h após 2 perdas
    # ✨ NOVO: Cooldown progressivo por perda
    "progressive_cooldown": True,
    "cooldown_multipliers": {
        1: 1.0,   # 1ª perda: 2h (120 min × 1.0)
        2: 2.0,   # 2ª perda: 4h (120 min × 2.0)
        3: 4.0,   # 3ª perda: 8h (120 min × 4.0) - bloqueia resto do dia
    }
    ,
    # ✅ Bloqueia o ativo pelo restante do dia após UM SL
    "block_full_day_on_single_sl": True
}

# ============================================
# 🔥 PRIORIDADE 2 - PIRÂMIDE INTELIGENTE
# ============================================

PYRAMID_REQUIREMENTS_ENHANCED = {
    **PYRAMID_REQUIREMENTS,  # Mantém configs antigas
    
    # ✅ NOVAS REGRAS CRÍTICAS
    "require_breakeven": True,  # SL no BE é OBRIGATÓRIO
    "require_1r_floating": True,  # OU ter +1R flutuante
    "min_time_between_legs_minutes": 15,  # Mín 15 min entre pernas
    "max_correlation_for_pyramid": 0.4,  # Não piramidar se correlação > 40%
}

# ============================================
# 🔥 PRIORIDADE 3 - LIMITE DIÁRIO POR ATIVO
# ============================================

DAILY_SYMBOL_LIMITS = {
    "enabled": True,
    "max_losing_trades_per_symbol": 1,  # Máx 2 perdas/ativo/dia
    "max_total_trades_per_symbol": 4,  # Máx 6 trades/ativo/dia (geral)
    "reset_time": "10:15",  # Reset junto com circuit breaker
}

# ============================================
# 🛡️ PROTEÇÃO ADICIONAL - VIX & DRAWDOWN (V5.2)
# ============================================

VIX_THRESHOLD_RISK_OFF = 35.0
VIX_THRESHOLD_PROTECTION = 25.0
MAX_PORTFOLIO_IBOV_CORR = 0.85
MAX_DAILY_DD_STOP = 0.05
MAX_SUBSETOR_EXPOSURE = 0.25 # 20% por subsetor
MIN_BOOK_IMBALANCE = 0.12
# ===========================
# LAND TRADING STRATEGY (V5.5)
# ===========================
ENABLE_BREAKEVEN = True
BREAKEVEN_ATR_MULT = 0.8  # Move para BE mais rápido (era 1.5)
ENABLE_TRAILING_STOP = True
ENABLE_PARTIAL_CLOSE = True
PARTIAL_CLOSE_ATR_MULT = 1.5 # Realiza parcial mais cedo
