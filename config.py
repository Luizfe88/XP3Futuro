import os
import json
import yaml
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
                'min_confidence': 0.65,  # Reduced from 0.70 for realistic ensemble performance
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
ML_MIN_CONFIDENCE = config_manager.get('ml.min_confidence', 0.65)  # Reduced from 0.78 to 0.65 for realistic ensemble
ML_MODEL_TYPE = config_manager.get('ml.model_type', 'ENSEMBLE')
ML_MIN_SAMPLES_FOR_RETRAIN = config_manager.get('ml.min_samples_for_retrain', 500)
ML_RETRAIN_THRESHOLD = 50  # Retreino reduzido para 50 trades
ML_Q_STATES = 5000  # Aumentado para 5000 estados
ML_TRAIN_PER_SYMBOL = config_manager.get('ml.train_per_symbol', True)
ML_PER_SYMBOL_MIN_SAMPLES = config_manager.get('ml.per_symbol_min_samples', 50)
WR_RESET_GRACE_TRADES = config_manager.get('risk.winrate_reset_grace_trades', 5)

# ✅ KELLY CRITERION & POSITION SIZING
KELLY_MULTIPLIER = 0.3  # Fractional Kelly (0.3x) for capital preservation
KELLY_MIN_TRADES_FOR_CALC = 30  # Minimum trades to calculate dynamic Kelly

# ✅ ATR VOLATILITY FILTERS
MAX_ATR_PCT = 5.0  # Base ATR limit (5.0% for blue chips)
ADAPTIVE_ATR_FILTER = True  # Enable adaptive ATR based on ADX
MAX_ATR_PCT_HIGH_ADX = 7.0  # Allow up to 7% ATR when ADX > 40 (strong trend)

# ===========================
# 🛡️ CONTROLES COMERCIAIS
# ===========================
MAX_TOTAL_EXPOSURE = int(config_manager.get('risk.max_total_exposure', 1_000_000))
TRADING_LUNCH_BREAK_START = "11:45"
TRADING_LUNCH_BREAK_END = "13:30"
MAX_SPREAD_TICKS = 4                   # Spread máximo permitido em ticks
DAILY_PROFIT_TARGET_PCT = 0.02          # Meta diária de lucro (2%)
BLOCK_AFTER_CONSECUTIVE_LOSSES = 2      # Parar após N perdas consecutivas
MARKET_HOURS_BUFFER_OPEN = 0           # Minutos após abertura para iniciar
MARKET_HOURS_BUFFER_CLOSE = 0          # Minutos antes do fechamento para parar
SLIPPAGE_ALERT_TICKS = 3                # Alertar se slippage > N ticks
MAX_VOL_THRESHOLD = 2.0
DAILY_VOLUME_LIMIT = 1000000000  # R$ 1 bilhão (limite financeiro diário)
# 🆕 LIMITES POR SUBSETOR (dentro de FINANCEIRO)
MAX_PER_SUBSETOR = {
    "BANCOS": 2,  # Máx 2 bancos (ITUB4, BBDC4, BBAS3, etc)
    "CORRETORAS": 1,  # Máx 1 corretora (B3SA3, BPAC11)
    "SEGUROS": 1,  # Máx 1 seguro (IRBR3, PSSA3)
}

# 🆕 Mapa de subsetores
SUBSETOR_MAP = {
    # BANCOS (máx 2)
    "ITUB4": "BANCOS", "BBDC4": "BANCOS", "BBDC3": "BANCOS", 
    "BBAS3": "BANCOS", "SANB11": "BANCOS", "BPAN4": "BANCOS",
    
    # CORRETORAS (máx 1)
    "B3SA3": "CORRETORAS", "BPAC11": "CORRETORAS",
    
    # SEGUROS (máx 1)
    "IRBR3": "SEGUROS", "PSSA3": "SEGUROS",
}

# Mapa de setores (todos os ativos monitorados)
SECTOR_MAP = {
    # FINANCEIRO (8/60): Bancos e serviços financeiros consolidados
    "ITUB4": "FINANCEIRO", "BBDC4": "FINANCEIRO", "BBAS3": "FINANCEIRO", "B3SA3": "FINANCEIRO",
    "BPAC11": "FINANCEIRO", "ITSA4": "FINANCEIRO", "ABCB4": "FINANCEIRO", "PINE4": "FINANCEIRO",

    # ENERGIA / UTILIDADES (8/60): Petróleo, energia elétrica e distribuição
    "PETR4": "ENERGIA", "PRIO3": "ENERGIA", "RECV3": "ENERGIA", "VBBR3": "ENERGIA",
    "AXIA3": "ENERGIA", "EQTL3": "ENERGIA", "ENEV3": "ENERGIA", "NEOE3": "ENERGIA",

    # MATERIAIS BÁSICOS (7/60): Mineração, siderurgia e papel/celulose
    "VALE3": "MATERIAIS BÁSICOS", "GGBR4": "MATERIAIS BÁSICOS", "USIM5": "MATERIAIS BÁSICOS",
    "CSNA3": "MATERIAIS BÁSICOS", "SUZB3": "MATERIAIS BÁSICOS", "KLBN11": "MATERIAIS BÁSICOS",
    "AURA33": "MATERIAIS BÁSICOS",

    # CONSUMO NÃO CÍCLICO (7/60): Alimentos, varejo essencial e agronegócio
    "ABEV3": "CONSUMO NÃO CÍCLICO", "JBSS3": "CONSUMO NÃO CÍCLICO", "BRFS3": "CONSUMO NÃO CÍCLICO",
    "BEEF3": "CONSUMO NÃO CÍCLICO", "CRFB3": "CONSUMO NÃO CÍCLICO", "SLCE3": "CONSUMO NÃO CÍCLICO",
    "RAIZ4": "CONSUMO NÃO CÍCLICO",

    # SAÚDE (6/60): Hospitais, planos e varejo farmacêutico
    "RDOR3": "SAÚDE", "HAPV3": "SAÚDE", "RADL3": "SAÚDE", "ONCO3": "SAÚDE",
    "QUAL3": "SAÚDE", "ANIM3": "SAÚDE",

    # CONSUMO CÍCLICO (8/60): Varejo, educação e construção civil
    "LREN3": "CONSUMO CÍCLICO", "MGLU3": "CONSUMO CÍCLICO", "YDUQ3": "CONSUMO CÍCLICO",
    "COGN3": "CONSUMO CÍCLICO", "CYRE3": "CONSUMO CÍCLICO", "MRVE3": "CONSUMO CÍCLICO",
    "TEND3": "CONSUMO CÍCLICO", "MDNE3": "CONSUMO CÍCLICO",

    # INDUSTRIAL (6/60): Logística, concessões e bens de capital
    "WEGE3": "INDUSTRIAL", "RENT3": "INDUSTRIAL", "MOVI3": "INDUSTRIAL",
    "RAIL3": "INDUSTRIAL", "CCRO3": "INDUSTRIAL", "AZUL4": "INDUSTRIAL",

    # TECNOLOGIA / COMUNICAÇÕES (5/60): Software e telecom
    "TOTS3": "TECNOLOGIA", "LWSA3": "TECNOLOGIA", "DESK3": "TECNOLOGIA",
    "VIVT3": "COMUNICAÇÕES", "TIMS3": "COMUNICAÇÕES",

    # IMOBILIÁRIO / DIVERSOS (5/60): Shoppings, saneamento e seguros
    "MULT3": "IMOBILIÁRIO", "IGTI11": "IMOBILIÁRIO",
    "SBSP3": "UTILIDADES", "BBSE3": "SEGUROS", "ODPV3": "SAÚDE"
    ,
    "WING26": "FUTUROS",
    "WDOG26": "FUTUROS",
    "SMALL$": "FUTUROS",
    "WSPH26": "FUTUROS",
    "BGIG26": "FUTUROS"
}


ACTIVE_FUTURES = {}


# Lista de símbolos proxy (usada em alguns módulos antigos - pode manter)
PROXY_SYMBOLS = [
    "VALE3",
    "PETR4",
    "ITUB4",
    "BBDC4",
    "BBAS3",
    "ABEV3",
    "WEGE3",
    "JBSS3",
    "RENT3",
    "PRIO3",
    "SUZB3",
    "AXIA3",
    "VIVT3",
    "HAPV3",
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
TRADING_START = "10:25"  # Após estabilização da abertura
NO_ENTRY_AFTER = "16:15"  # Fim das entradas (antes do fechamento nervoso)
CLOSE_ALL_BY = "16:40"  # FECHAMENTO FORÇADO (nunca posar no after)
NO_ENTRY_BEFORE_CLOSE_MINUTES = 15  # Bloqueia novas entradas quando faltar pouco p/ fechar
DAILY_RESET_TIME = "00:00"  # Reset diário do circuit breaker
DAY_ONLY_MODE = True
FRIDAY_NO_ENTRY_AFTER = "15:30"
FRIDAY_CLOSE_ALL_BY = "15:30"
FRIDAY_NO_ENTRY_BEFORE_CLOSE_MINUTES = 15
FRIDAY_MARKET_HOURS_BUFFER_CLOSE = 0
FRIDAY_RISK_REDUCE_AFTER = "14:30"
FRIDAY_RISK_FACTOR_MULT = 0.60
FRIDAY_DISABLE_PYRAMID_AFTER = "14:30"
# ✅ Pausa operacional entre 11:45 e 13:30
TRADING_LUNCH_BREAK_START = "11:45"
TRADING_LUNCH_BREAK_END = "13:30"
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
    {"start": "09:55", "end": "10:20", "label": "Abertura (volatilidade e leilão)"},
    {"start": "14:25", "end": "14:40", "label": "Macro EUA (horário 14:30)"},
    {"start": "15:55", "end": "16:10", "label": "Macro BR (horário 16:00)"},
]
ENABLE_NEWS_SENTIMENT_BLOCK = False
NEWS_SENTIMENT_NEG_THRESHOLD = -0.70
NEWS_SENTIMENT_BLOCK_MINUTES = 60

ML_MODE = config_manager.get('ml.mode', 'advisory')  # advisory|gate
ML_ADVISORY_HARD_BLOCK = config_manager.get('ml.advisory_hard_block', 0.82)
ML_ADVISORY_SOFT_RISK = config_manager.get('ml.advisory_soft_risk', 0.70)
ML_ADVISORY_SOFT_RISK_FACTOR = config_manager.get('ml.advisory_soft_risk_factor', 0.60)

ENTRY_SCORE_DELTA_MORNING = config_manager.get('entry.score_delta_morning', 5)
ENTRY_SCORE_DELTA_LUNCH = config_manager.get('entry.score_delta_lunch', 10)
ENTRY_SCORE_DELTA_AFTERNOON = config_manager.get('entry.score_delta_afternoon', 0)
MAX_RISK_PER_SYMBOL_PCT = 0.02  # Máximo 2% da equity por papel
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
MIN_RR_FUTURES = 2.5
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
MIN_RR = 2.5                         # R:R mínimo considerando custos
DAILY_STOP_MONEY = 2000              # Stop diário absoluto em R$
ML_MIN_TRADES_ENABLE = 200           # ML desativado até 200 trades

# NOVOS: A/B Testing e Infra
AB_TEST_ENABLED = True
AB_TEST_GROUPS = {
    "A": {"min_confidence": 0.61, "signal_threshold": 61},
    "B": {"min_confidence": 0.65, "signal_threshold": 65}
}
REDIS_CACHE_TTL_TICK = 1    # TTL em segundos para ticks
REDIS_CACHE_TTL_INFO = 60   # TTL em segundos para symbol_info

# =========================================================
# 🌐 POLYGON.IO API (DADOS B3)
# =========================================================
POLYGON_API_KEY = "xrE09LEWJYBZfQcV57pCvsw4aqkOiqbz"
POLYGON_BASE_URL = "https://api.polygon.io"
POLYGON_CACHE_TTL = 30  # Cache de 30 segundos para dados Polygon

NEWS_API_KEY = "c39162901d1a45eeaad80d3c3f6f8c1e"  # NewsAPI.org


# Slippage realista B3 (por liquidez/spread)
SLIPPAGE_MAP = {
    # Alta liquidez (top 10 volume B3)
    "PETR4": 0.0010,
    "VALE3": 0.0010,
    "ITUB4": 0.0012,
    "BBDC4": 0.0012,
    "BBAS3": 0.0013,
    "ABEV3": 0.0015,
    # Média liquidez (80% do SECTOR_MAP)
    "DEFAULT": 0.0030,  # 0.30% base
}
MAX_SPREAD_ACTION_PCT = 0.30
MAX_SPREAD_FUTURE_POINTS = 20
ACTION_COST_PCT = 0.00055

# =========================================================
# 📊 PARÂMETROS DINÂMICOS POR WIN RATE
# =========================================================
# Ajusta agressividade baseado no win rate atual

WIN_RATE_TIERS = {
    # Win Rate >= 70%: Modo agressivo
    "HIGH": {
        "min_win_rate": 0.70,
        "kelly_multiplier": 0.15,      # Kelly 15%
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
        "kelly_multiplier": 0.08,
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
SL_ATR_MULTIPLIER = 2.0  # SL inicial = preço ± ATR × 2.0
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
# ===========================
# NOTIFICAÇÕES TELEGRAM
# ===========================
ENABLE_TELEGRAM_NOTIF = True
TELEGRAM_BOT_TOKEN = (
    "8551934559:AAGZRMxH51N-IcsAuFJzelafOuVo1pMS9nI"  # Ex: 123456789:AAF...
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
        "adx_min": 25,
        "min_score": 40,
        "atr_max": 8.0,
        "min_volume_ratio": 1.1,     # Volume atual > 130% da média de 20 períodos
        "require_vwap_proximity": True,  # Preço perto do VWAP intraday (±1%)
        "min_momentum": 0.0007,  # Momentum mínimo mais exigente
    },
    "MID": {
        "start": "11:30",
        "end": "14:30",
        "adx_min": 25,
        "min_score": 35,
        "atr_max": 10.0,
        "min_volume_ratio": 1.05
    },
    "LATE": {
        "start": "14:30",
        "end": "16:55",
        "adx_min": 25,
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
 "B3SA3": {
            "ema_short": 17,
            "ema_long": 37,
            "rsi_low": 28,
            "rsi_high": 66,
            "adx_threshold": 27,
            "sl_atr_multiplier": 1.8,
            "tp_ratio": 3.0,
            "weight": 0.05
        },
        "ABEV3": {
            "ema_short": 20,
            "ema_long": 71,
            "rsi_low": 40,
            "rsi_high": 60,
            "adx_threshold": 29,
            "sl_atr_multiplier": 1.5,
            "tp_ratio": 2.6,
            "weight": 0.09
        },
        "PETR4": {
            "ema_short": 11,
            "ema_long": 56,
            "rsi_low": 40,
            "rsi_high": 60,
            "adx_threshold": 29,
            "sl_atr_multiplier": 1.5,
            "tp_ratio": 2.6,
            "weight": 0.08
        },
        "VALE3": {
            "ema_short": 12,
            "ema_long": 97,
            "rsi_low": 27,
            "rsi_high": 64,
            "adx_threshold": 18,
            "sl_atr_multiplier": 2.4,
            "tp_ratio": 2.0,
            "weight": 0.08
        },
        "BBDC4": {
            "ema_short": 14,
            "ema_long": 58,
            "rsi_low": 40,
            "rsi_high": 60,
            "adx_threshold": 29,
            "sl_atr_multiplier": 1.5,
            "tp_ratio": 2.6,
            "weight": 0.07
        },
        "BEEF3": {
            "ema_short": 15,
            "ema_long": 60,
            "rsi_low": 36,
            "rsi_high": 67,
            "adx_threshold": 18,
            "sl_atr_multiplier": 3.3,
            "tp_ratio": 3.0,
            "weight": 0.10
        },
        "SUZB3": {
            "ema_short": 10,
            "ema_long": 65,
            "rsi_low": 28,
            "rsi_high": 72,
            "adx_threshold": 32,
            "sl_atr_multiplier": 2.7,
            "tp_ratio": 2.0,
            "weight": 0.09
        },
        "ITUB4": {
            "ema_short": 16,
            "ema_long": 97,
            "rsi_low": 36,
            "rsi_high": 72,
            "adx_threshold": 18,
            "sl_atr_multiplier": 1.8,
            "tp_ratio": 1.2,
            "weight": 0.09
        },
        "SBSP3": {
            "ema_short": 8,
            "ema_long": 82,
            "rsi_low": 38,
            "rsi_high": 75,
            "adx_threshold": 26,
            "sl_atr_multiplier": 2.6,
            "tp_ratio": 2.8,
            "weight": 0.12
        },
        "VIVT3": {
            "ema_short": 21,
            "ema_long": 38,
            "rsi_low": 34,
            "rsi_high": 63,
            "adx_threshold": 16,
            "sl_atr_multiplier": 3.4,
            "tp_ratio": 3.0,
            "weight": 0.09
        },
        "WEGE3": {
            "ema_short": 18,
            "ema_long": 59,
            "rsi_low": 38,
            "rsi_high": 65,
            "adx_threshold": 19,
            "sl_atr_multiplier": 1.5,
            "tp_ratio": 1.8,
            "weight": 0.12
        },
        "TOTS3": {
            "ema_short": 18,
            "ema_long": 46,
            "rsi_low": 39,
            "rsi_high": 66,
            "adx_threshold": 18,
            "sl_atr_multiplier": 2.0,
            "tp_ratio": 2.0,
            "weight": 0.11
        },
           
        "AURA33": {
            "ema_short": 24,
            "ema_long": 139,
            "rsi_low": 29,
            "rsi_high": 71,
            "adx_threshold": 33,
            "sl_atr_multiplier": 3.2,
            "tp_ratio": 2.8,
            "weight": 0.06
        },
"WING26": {
        "ema_short": 8,
        "ema_long": 47,
        "adx_threshold": 12,
        "rsi_low": 40,
        "rsi_high": 64,
        "sl_atr_multiplier": 1.08,
        "tp_ratio": 2.55,
        "weight": 0.35
    },
    "WDOG26": {
        "ema_short": 8,
        "ema_long": 44,
        "adx_threshold": 10,
        "rsi_low": 38,
        "rsi_high": 61,
        "sl_atr_multiplier": 1.05,
        "tp_ratio": 2.54,
        "weight": 0.25
    },
    "WSPH26": {
        "ema_short": 14,
        "ema_long": 48,
        "adx_threshold": 17,
        "rsi_low": 40,
        "rsi_high": 68,
        "sl_atr_multiplier": 1.15,
        "tp_ratio": 3.14,
        "weight": 0.30
    },
    "BGIG26": {
        "ema_short": 12,
        "ema_long": 47,
        "adx_threshold": 11,
        "rsi_low": 39,
        "rsi_high": 68,
        "sl_atr_multiplier": 2.50,
        "tp_ratio": 1.80,
        "weight": 0.10
    }
    }

ELITE_BLUE_CHIPS = ["PETR4", "VALE3", "ITUB4", "BBDC4", "ABEV3", "BBAS3", "B3SA3", "VIVT3", "AXIA3", "SUZB3"]

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

ATIVOS_VIAVEIS_REPORT_JSON = r"""{
  "metadados": {
    "timestamp": "2026-01-17T14:53:42.370123",
    "versao_sistema": "XP3 PRO v7.0"
  },
  "sumario_executivo": {
    "total_avaliados": 55,
    "total_viaveis": 27,
    "percentual_viabilidade": 0.4909090909090909,
    "distribuicao_por_categorias": {
      "OPORTUNIDADE": 18,
      "BLUE CHIP": 9
    },
    "principais_oportunidades": [
      {
        "symbol": "MULT3",
        "tier": "B",
        "score_total": 0.5854545454545454,
        "avg_fin_volume": 441644.89197500004,
        "volatility_ann": 0.24363547531556587,
        "abs_corr_ibov": 0.803115315476122,
        "risk_notes": "Correlação alta com IBOV reduz diversificação; monitorar em dias de stress."
      },
      {
        "symbol": "LREN3",
        "tier": "B",
        "score_total": 0.5472727272727272,
        "avg_fin_volume": 318888.73997500003,
        "volatility_ann": 0.36318340080298694,
        "abs_corr_ibov": 0.6372665915113427,
        "risk_notes": "Correlação alta com IBOV reduz diversificação; monitorar em dias de stress."
      },
      {
        "symbol": "ENEV3",
        "tier": "C",
        "score_total": 0.5336363636363636,
        "avg_fin_volume": 357555.44850000006,
        "volatility_ann": 0.24590499672757524,
        "abs_corr_ibov": 0.6411394246550437,
        "risk_notes": "Correlação alta com IBOV reduz diversificação; monitorar em dias de stress."
      },
      {
        "symbol": "CYRE3",
        "tier": "C",
        "score_total": 0.5272727272727272,
        "avg_fin_volume": 444719.80100000004,
        "volatility_ann": 0.33474525930223953,
        "abs_corr_ibov": 0.7247709490549927,
        "risk_notes": "Correlação alta com IBOV reduz diversificação; monitorar em dias de stress."
      },
      {
        "symbol": "EQTL3",
        "tier": "C",
        "score_total": 0.51,
        "avg_fin_volume": 794208.4575250001,
        "volatility_ann": 0.22173657044312492,
        "abs_corr_ibov": 0.817097851068095,
        "risk_notes": "Correlação alta com IBOV reduz diversificação; monitorar em dias de stress."
      },
      {
        "symbol": "VBBR3",
        "tier": "C",
        "score_total": 0.5072727272727273,
        "avg_fin_volume": 451809.75815000007,
        "volatility_ann": 0.25523327806973445,
        "abs_corr_ibov": 0.6328859876490274,
        "risk_notes": "Correlação alta com IBOV reduz diversificação; monitorar em dias de stress."
      },
      {
        "symbol": "GGBR4",
        "tier": "C",
        "score_total": 0.5063636363636363,
        "avg_fin_volume": 331275.375625,
        "volatility_ann": 0.2195128702649207,
        "abs_corr_ibov": 0.4386512813767289,
        "risk_notes": "Correlação alta com IBOV reduz diversificação; monitorar em dias de stress."
      },
      {
        "symbol": "BBSE3",
        "tier": "C",
        "score_total": 0.4845454545454545,
        "avg_fin_volume": 417630.247975,
        "volatility_ann": 0.1532896315643996,
        "abs_corr_ibov": 0.4270386794176928,
        "risk_notes": "Correlação alta com IBOV reduz diversificação; monitorar em dias de stress."
      },
      {
        "symbol": "VIVT3",
        "tier": "C",
        "score_total": 0.4790909090909091,
        "avg_fin_volume": 352506.52770000004,
        "volatility_ann": 0.21559613920074364,
        "abs_corr_ibov": 0.5285535671410724,
        "risk_notes": "Correlação alta com IBOV reduz diversificação; monitorar em dias de stress."
      },
      {
        "symbol": "TIMS3",
        "tier": "C",
        "score_total": 0.4727272727272727,
        "avg_fin_volume": 349537.802,
        "volatility_ann": 0.20838919569682463,
        "abs_corr_ibov": 0.41481586592251674,
        "risk_notes": "Correlação alta com IBOV reduz diversificação; monitorar em dias de stress."
      }
    ],
    "principais_riscos": [
      "Dependência de dados (MT5 ou Yahoo). Yahoo M15 é limitado (~60 dias).",
      "Market cap pode estar ausente (0) para alguns ativos e reduzir confiança do filtro.",
      "Correlação com IBOV pode concentrar risco sistêmico; considerar balanceamento por setor."
    ]
  },
  "ativos_viaveis": [
    {
      "identificacao": {
        "nome": "MULT3",
        "codigo": "MULT3",
        "codigo_unico": "MULT3",
        "tipo": "AÇÃO",
        "categoria": "OPORTUNIDADE"
      },
      "ativo": {
        "category": "OPORTUNIDADE",
        "weight": 0.05,
        "ema_short": 9,
        "ema_long": 21,
        "rsi_low": 30.0,
        "rsi_high": 70.0,
        "adx_threshold": 25.0,
        "mom_min": 0.0,
        "sl_atr_multiplier": 2.5,
        "tp_mult": 5.0
      },
      "criterios_viabilidade": {
        "liquidez": {
          "metrica": "avg_fin_volume",
          "valor": 441644.89197500004,
          "limiar": 318888.73997500003,
          "condicao": ">= limiar",
          "ok": true
        },
        "volatilidade": {
          "metrica": "volatility_ann",
          "valor": 0.24363547531556587,
          "limiar_inferior": 0.21543975635038212,
          "limiar_superior": 0.5751659600971757,
          "condicao": "entre [limiar_inferior, limiar_superior]",
          "ok": true
        },
        "correlacao_ibov": {
          "metrica": "abs_corr_ibov",
          "valor": 0.803115315476122,
          "limiar": 0.6353184419932515,
          "condicao": "<= limiar",
          "ok": false
        },
        "market_cap": {
          "metrica": "market_cap",
          "valor": 14239464448.0,
          "limiar": 6132724479.999999,
          "condicao": ">= limiar (quando limiar > 0)",
          "ok": true
        },
        "regra_final": {
          "blue_chip": false,
          "criteria_passed": 3,
          "liquidez_obrigatoria": true,
          "min_criterios": 3
        }
      },
      "resultados": {
        "rank_total": 18,
        "tier": "B",
        "score_total": 0.5854545454545454,
        "liquidez_avg_fin_volume": 441644.89197500004,
        "volatilidade_anualizada": 0.24363547531556587,
        "correlacao_ibov": 0.803115315476122,
        "market_cap": 14239464448.0,
        "fonte_dados": "mt5"
      },
      "recomendacoes": {
        "acao_recomendada": "Adicionar ao universo de execução da próxima semana",
        "modo": "OPORTUNIDADE",
        "parametrizacao": "Usar ELITE_SYMBOLS (se existir) ou DEFAULT_PARAMS como base e refinar via otimização WFO.",
        "monitoramento": "Monitorar volatilidade e correlação diária; ajustar exposição se |corr| subir."
      },
      "prazos_e_recursos": {
        "prazo_estimado": "1 dia útil",
        "recursos_necessarios": [
          "MT5 conectado",
          "dados D1/M15",
          "cache Yahoo habilitado",
          "tempo de backtest"
        ]
      },
      "restricoes_dependencias": {
        "dependencias": [
          "MetaTrader5",
          "yfinance"
        ],
        "observacoes": [
          "Se o MT5 não tiver o símbolo, o script usa Yahoo como fallback.",
          "Backtest M15 com Yahoo é limitado e deve ser validado com dados do broker quando possível."
        ]
      },
      "justificativa_tecnica": "Liquidez OK (R$ 441.645; lim=R$ 318.889); Vol OK (24,36%; faixa=21,54%–57,52%); Corr NOK (|corr|=0,803; lim=0,635); MCap OK (R$ 14.239.464.448; lim=R$ 6.132.724.480)"
    },
    {
      "identificacao": {
        "nome": "LREN3",
        "codigo": "LREN3",
        "codigo_unico": "LREN3",
        "tipo": "AÇÃO",
        "categoria": "OPORTUNIDADE"
      },
      "ativo": {
        "category": "OPORTUNIDADE",
        "weight": 0.0407,
        "ema_short": 25,
        "ema_long": 84,
        "rsi_low": 38,
        "rsi_high": 85,
        "adx_threshold": 24,
        "mom_min": 0.0,
        "sl_atr_multiplier": 3.5,
        "tp_mult": 3.0
      },
      "criterios_viabilidade": {
        "liquidez": {
          "metrica": "avg_fin_volume",
          "valor": 318888.73997500003,
          "limiar": 318888.73997500003,
          "condicao": ">= limiar",
          "ok": true
        },
        "volatilidade": {
          "metrica": "volatility_ann",
          "valor": 0.36318340080298694,
          "limiar_inferior": 0.21543975635038212,
          "limiar_superior": 0.5751659600971757,
          "condicao": "entre [limiar_inferior, limiar_superior]",
          "ok": true
        },
        "correlacao_ibov": {
          "metrica": "abs_corr_ibov",
          "valor": 0.6372665915113427,
          "limiar": 0.6353184419932515,
          "condicao": "<= limiar",
          "ok": false
        },
        "market_cap": {
          "metrica": "market_cap",
          "valor": 13092715520.0,
          "limiar": 6132724479.999999,
          "condicao": ">= limiar (quando limiar > 0)",
          "ok": true
        },
        "regra_final": {
          "blue_chip": false,
          "criteria_passed": 3,
          "liquidez_obrigatoria": true,
          "min_criterios": 3
        }
      },
      "resultados": {
        "rank_total": 23,
        "tier": "B",
        "score_total": 0.5472727272727272,
        "liquidez_avg_fin_volume": 318888.73997500003,
        "volatilidade_anualizada": 0.36318340080298694,
        "correlacao_ibov": 0.6372665915113427,
        "market_cap": 13092715520.0,
        "fonte_dados": "mt5"
      },
      "recomendacoes": {
        "acao_recomendada": "Adicionar ao universo de execução da próxima semana",
        "modo": "OPORTUNIDADE",
        "parametrizacao": "Usar ELITE_SYMBOLS (se existir) ou DEFAULT_PARAMS como base e refinar via otimização WFO.",
        "monitoramento": "Monitorar volatilidade e correlação diária; ajustar exposição se |corr| subir."
      },
      "prazos_e_recursos": {
        "prazo_estimado": "1 dia útil",
        "recursos_necessarios": [
          "MT5 conectado",
          "dados D1/M15",
          "cache Yahoo habilitado",
          "tempo de backtest"
        ]
      },
      "restricoes_dependencias": {
        "dependencias": [
          "MetaTrader5",
          "yfinance"
        ],
        "observacoes": [
          "Se o MT5 não tiver o símbolo, o script usa Yahoo como fallback.",
          "Backtest M15 com Yahoo é limitado e deve ser validado com dados do broker quando possível."
        ]
      },
      "justificativa_tecnica": "Liquidez OK (R$ 318.889; lim=R$ 318.889); Vol OK (36,32%; faixa=21,54%–57,52%); Corr NOK (|corr|=0,637; lim=0,635); MCap OK (R$ 13.092.715.520; lim=R$ 6.132.724.480)"
    },
    {
      "identificacao": {
        "nome": "ENEV3",
        "codigo": "ENEV3",
        "codigo_unico": "ENEV3",
        "tipo": "AÇÃO",
        "categoria": "OPORTUNIDADE"
      },
      "ativo": {
        "category": "OPORTUNIDADE",
        "weight": 0.0936,
        "ema_short": 23,
        "ema_long": 72,
        "rsi_low": 36,
        "rsi_high": 55,
        "adx_threshold": 30,
        "mom_min": 0.0,
        "sl_atr_multiplier": 4.1,
        "tp_mult": 3.0
      },
      "criterios_viabilidade": {
        "liquidez": {
          "metrica": "avg_fin_volume",
          "valor": 357555.44850000006,
          "limiar": 318888.73997500003,
          "condicao": ">= limiar",
          "ok": true
        },
        "volatilidade": {
          "metrica": "volatility_ann",
          "valor": 0.24590499672757524,
          "limiar_inferior": 0.21543975635038212,
          "limiar_superior": 0.5751659600971757,
          "condicao": "entre [limiar_inferior, limiar_superior]",
          "ok": true
        },
        "correlacao_ibov": {
          "metrica": "abs_corr_ibov",
          "valor": 0.6411394246550437,
          "limiar": 0.6353184419932515,
          "condicao": "<= limiar",
          "ok": false
        },
        "market_cap": {
          "metrica": "market_cap",
          "valor": 39465570304.0,
          "limiar": 6132724479.999999,
          "condicao": ">= limiar (quando limiar > 0)",
          "ok": true
        },
        "regra_final": {
          "blue_chip": false,
          "criteria_passed": 3,
          "liquidez_obrigatoria": true,
          "min_criterios": 3
        }
      },
      "resultados": {
        "rank_total": 26,
        "tier": "C",
        "score_total": 0.5336363636363636,
        "liquidez_avg_fin_volume": 357555.44850000006,
        "volatilidade_anualizada": 0.24590499672757524,
        "correlacao_ibov": 0.6411394246550437,
        "market_cap": 39465570304.0,
        "fonte_dados": "mt5"
      },
      "recomendacoes": {
        "acao_recomendada": "Adicionar ao universo de execução da próxima semana",
        "modo": "OPORTUNIDADE",
        "parametrizacao": "Usar ELITE_SYMBOLS (se existir) ou DEFAULT_PARAMS como base e refinar via otimização WFO.",
        "monitoramento": "Monitorar volatilidade e correlação diária; ajustar exposição se |corr| subir."
      },
      "prazos_e_recursos": {
        "prazo_estimado": "1 dia útil",
        "recursos_necessarios": [
          "MT5 conectado",
          "dados D1/M15",
          "cache Yahoo habilitado",
          "tempo de backtest"
        ]
      },
      "restricoes_dependencias": {
        "dependencias": [
          "MetaTrader5",
          "yfinance"
        ],
        "observacoes": [
          "Se o MT5 não tiver o símbolo, o script usa Yahoo como fallback.",
          "Backtest M15 com Yahoo é limitado e deve ser validado com dados do broker quando possível."
        ]
      },
      "justificativa_tecnica": "Liquidez OK (R$ 357.555; lim=R$ 318.889); Vol OK (24,59%; faixa=21,54%–57,52%); Corr NOK (|corr|=0,641; lim=0,635); MCap OK (R$ 39.465.570.304; lim=R$ 6.132.724.480)"
    },
    {
      "identificacao": {
        "nome": "CYRE3",
        "codigo": "CYRE3",
        "codigo_unico": "CYRE3",
        "tipo": "AÇÃO",
        "categoria": "OPORTUNIDADE"
      },
      "ativo": {
        "category": "OPORTUNIDADE",
        "weight": 0.0575,
        "ema_short": 22,
        "ema_long": 44,
        "rsi_low": 35,
        "rsi_high": 75,
        "adx_threshold": 16,
        "mom_min": 0.0,
        "sl_atr_multiplier": 4.1,
        "tp_mult": 3.0
      },
      "criterios_viabilidade": {
        "liquidez": {
          "metrica": "avg_fin_volume",
          "valor": 444719.80100000004,
          "limiar": 318888.73997500003,
          "condicao": ">= limiar",
          "ok": true
        },
        "volatilidade": {
          "metrica": "volatility_ann",
          "valor": 0.33474525930223953,
          "limiar_inferior": 0.21543975635038212,
          "limiar_superior": 0.5751659600971757,
          "condicao": "entre [limiar_inferior, limiar_superior]",
          "ok": true
        },
        "correlacao_ibov": {
          "metrica": "abs_corr_ibov",
          "valor": 0.7247709490549927,
          "limiar": 0.6353184419932515,
          "condicao": "<= limiar",
          "ok": false
        },
        "market_cap": {
          "metrica": "market_cap",
          "valor": 9047880704.0,
          "limiar": 6132724479.999999,
          "condicao": ">= limiar (quando limiar > 0)",
          "ok": true
        },
        "regra_final": {
          "blue_chip": false,
          "criteria_passed": 3,
          "liquidez_obrigatoria": true,
          "min_criterios": 3
        }
      },
      "resultados": {
        "rank_total": 27,
        "tier": "C",
        "score_total": 0.5272727272727272,
        "liquidez_avg_fin_volume": 444719.80100000004,
        "volatilidade_anualizada": 0.33474525930223953,
        "correlacao_ibov": 0.7247709490549927,
        "market_cap": 9047880704.0,
        "fonte_dados": "mt5"
      },
      "recomendacoes": {
        "acao_recomendada": "Adicionar ao universo de execução da próxima semana",
        "modo": "OPORTUNIDADE",
        "parametrizacao": "Usar ELITE_SYMBOLS (se existir) ou DEFAULT_PARAMS como base e refinar via otimização WFO.",
        "monitoramento": "Monitorar volatilidade e correlação diária; ajustar exposição se |corr| subir."
      },
      "prazos_e_recursos": {
        "prazo_estimado": "1 dia útil",
        "recursos_necessarios": [
          "MT5 conectado",
          "dados D1/M15",
          "cache Yahoo habilitado",
          "tempo de backtest"
        ]
      },
      "restricoes_dependencias": {
        "dependencias": [
          "MetaTrader5",
          "yfinance"
        ],
        "observacoes": [
          "Se o MT5 não tiver o símbolo, o script usa Yahoo como fallback.",
          "Backtest M15 com Yahoo é limitado e deve ser validado com dados do broker quando possível."
        ]
      },
      "justificativa_tecnica": "Liquidez OK (R$ 444.720; lim=R$ 318.889); Vol OK (33,47%; faixa=21,54%–57,52%); Corr NOK (|corr|=0,725; lim=0,635); MCap OK (R$ 9.047.880.704; lim=R$ 6.132.724.480)"
    },
    {
      "identificacao": {
        "nome": "EQTL3",
        "codigo": "EQTL3",
        "codigo_unico": "EQTL3",
        "tipo": "AÇÃO",
        "categoria": "OPORTUNIDADE"
      },
      "ativo": {
        "category": "OPORTUNIDADE",
        "weight": 0.0754,
        "ema_short": 19,
        "ema_long": 60,
        "rsi_low": 33,
        "rsi_high": 64,
        "adx_threshold": 22,
        "mom_min": 0.0,
        "sl_atr_multiplier": 4.1,
        "tp_mult": 3.0
      },
      "criterios_viabilidade": {
        "liquidez": {
          "metrica": "avg_fin_volume",
          "valor": 794208.4575250001,
          "limiar": 318888.73997500003,
          "condicao": ">= limiar",
          "ok": true
        },
        "volatilidade": {
          "metrica": "volatility_ann",
          "valor": 0.22173657044312492,
          "limiar_inferior": 0.21543975635038212,
          "limiar_superior": 0.5751659600971757,
          "condicao": "entre [limiar_inferior, limiar_superior]",
          "ok": true
        },
        "correlacao_ibov": {
          "metrica": "abs_corr_ibov",
          "valor": 0.817097851068095,
          "limiar": 0.6353184419932515,
          "condicao": "<= limiar",
          "ok": false
        },
        "market_cap": {
          "metrica": "market_cap",
          "valor": 47108534272.0,
          "limiar": 6132724479.999999,
          "condicao": ">= limiar (quando limiar > 0)",
          "ok": true
        },
        "regra_final": {
          "blue_chip": false,
          "criteria_passed": 3,
          "liquidez_obrigatoria": true,
          "min_criterios": 3
        }
      },
      "resultados": {
        "rank_total": 28,
        "tier": "C",
        "score_total": 0.51,
        "liquidez_avg_fin_volume": 794208.4575250001,
        "volatilidade_anualizada": 0.22173657044312492,
        "correlacao_ibov": 0.817097851068095,
        "market_cap": 47108534272.0,
        "fonte_dados": "mt5"
      },
      "recomendacoes": {
        "acao_recomendada": "Adicionar ao universo de execução da próxima semana",
        "modo": "OPORTUNIDADE",
        "parametrizacao": "Usar ELITE_SYMBOLS (se existir) ou DEFAULT_PARAMS como base e refinar via otimização WFO.",
        "monitoramento": "Monitorar volatilidade e correlação diária; ajustar exposição se |corr| subir."
      },
      "prazos_e_recursos": {
        "prazo_estimado": "1 dia útil",
        "recursos_necessarios": [
          "MT5 conectado",
          "dados D1/M15",
          "cache Yahoo habilitado",
          "tempo de backtest"
        ]
      },
      "restricoes_dependencias": {
        "dependencias": [
          "MetaTrader5",
          "yfinance"
        ],
        "observacoes": [
          "Se o MT5 não tiver o símbolo, o script usa Yahoo como fallback.",
          "Backtest M15 com Yahoo é limitado e deve ser validado com dados do broker quando possível."
        ]
      },
      "justificativa_tecnica": "Liquidez OK (R$ 794.208; lim=R$ 318.889); Vol OK (22,17%; faixa=21,54%–57,52%); Corr NOK (|corr|=0,817; lim=0,635); MCap OK (R$ 47.108.534.272; lim=R$ 6.132.724.480)"
    },
    {
      "identificacao": {
        "nome": "VBBR3",
        "codigo": "VBBR3",
        "codigo_unico": "VBBR3",
        "tipo": "AÇÃO",
        "categoria": "OPORTUNIDADE"
      },
      "ativo": {
        "category": "OPORTUNIDADE",
        "weight": 0.0482,
        "ema_short": 19,
        "ema_long": 40,
        "rsi_low": 39,
        "rsi_high": 70,
        "adx_threshold": 14,
        "mom_min": 0.0,
        "sl_atr_multiplier": 3.5,
        "tp_mult": 3.0
      },
      "criterios_viabilidade": {
        "liquidez": {
          "metrica": "avg_fin_volume",
          "valor": 451809.75815000007,
          "limiar": 318888.73997500003,
          "condicao": ">= limiar",
          "ok": true
        },
        "volatilidade": {
          "metrica": "volatility_ann",
          "valor": 0.25523327806973445,
          "limiar_inferior": 0.21543975635038212,
          "limiar_superior": 0.5751659600971757,
          "condicao": "entre [limiar_inferior, limiar_superior]",
          "ok": true
        },
        "correlacao_ibov": {
          "metrica": "abs_corr_ibov",
          "valor": 0.6328859876490274,
          "limiar": 0.6353184419932515,
          "condicao": "<= limiar",
          "ok": true
        },
        "market_cap": {
          "metrica": "market_cap",
          "valor": 30564724736.0,
          "limiar": 6132724479.999999,
          "condicao": ">= limiar (quando limiar > 0)",
          "ok": true
        },
        "regra_final": {
          "blue_chip": false,
          "criteria_passed": 4,
          "liquidez_obrigatoria": true,
          "min_criterios": 3
        }
      },
      "resultados": {
        "rank_total": 30,
        "tier": "C",
        "score_total": 0.5072727272727273,
        "liquidez_avg_fin_volume": 451809.75815000007,
        "volatilidade_anualizada": 0.25523327806973445,
        "correlacao_ibov": 0.6328859876490274,
        "market_cap": 30564724736.0,
        "fonte_dados": "mt5"
      },
      "recomendacoes": {
        "acao_recomendada": "Adicionar ao universo de execução da próxima semana",
        "modo": "OPORTUNIDADE",
        "parametrizacao": "Usar ELITE_SYMBOLS (se existir) ou DEFAULT_PARAMS como base e refinar via otimização WFO.",
        "monitoramento": "Monitorar volatilidade e correlação diária; ajustar exposição se |corr| subir."
      },
      "prazos_e_recursos": {
        "prazo_estimado": "1 dia útil",
        "recursos_necessarios": [
          "MT5 conectado",
          "dados D1/M15",
          "cache Yahoo habilitado",
          "tempo de backtest"
        ]
      },
      "restricoes_dependencias": {
        "dependencias": [
          "MetaTrader5",
          "yfinance"
        ],
        "observacoes": [
          "Se o MT5 não tiver o símbolo, o script usa Yahoo como fallback.",
          "Backtest M15 com Yahoo é limitado e deve ser validado com dados do broker quando possível."
        ]
      },
      "justificativa_tecnica": "Liquidez OK (R$ 451.810; lim=R$ 318.889); Vol OK (25,52%; faixa=21,54%–57,52%); Corr OK (|corr|=0,633; lim=0,635); MCap OK (R$ 30.564.724.736; lim=R$ 6.132.724.480)"
    },
    {
      "identificacao": {
        "nome": "GGBR4",
        "codigo": "GGBR4",
        "codigo_unico": "GGBR4",
        "tipo": "AÇÃO",
        "categoria": "OPORTUNIDADE"
      },
      "ativo": {
        "category": "OPORTUNIDADE",
        "weight": 0.0157,
        "ema_short": 20,
        "ema_long": 86,
        "rsi_low": 33,
        "rsi_high": 56,
        "adx_threshold": 28,
        "mom_min": 0.0,
        "sl_atr_multiplier": 4.1,
        "tp_mult": 3.0
      },
      "criterios_viabilidade": {
        "liquidez": {
          "metrica": "avg_fin_volume",
          "valor": 331275.375625,
          "limiar": 318888.73997500003,
          "condicao": ">= limiar",
          "ok": true
        },
        "volatilidade": {
          "metrica": "volatility_ann",
          "valor": 0.2195128702649207,
          "limiar_inferior": 0.21543975635038212,
          "limiar_superior": 0.5751659600971757,
          "condicao": "entre [limiar_inferior, limiar_superior]",
          "ok": true
        },
        "correlacao_ibov": {
          "metrica": "abs_corr_ibov",
          "valor": 0.4386512813767289,
          "limiar": 0.6353184419932515,
          "condicao": "<= limiar",
          "ok": true
        },
        "market_cap": {
          "metrica": "market_cap",
          "valor": 43532173312.0,
          "limiar": 6132724479.999999,
          "condicao": ">= limiar (quando limiar > 0)",
          "ok": true
        },
        "regra_final": {
          "blue_chip": false,
          "criteria_passed": 4,
          "liquidez_obrigatoria": true,
          "min_criterios": 3
        }
      },
      "resultados": {
        "rank_total": 31,
        "tier": "C",
        "score_total": 0.5063636363636363,
        "liquidez_avg_fin_volume": 331275.375625,
        "volatilidade_anualizada": 0.2195128702649207,
        "correlacao_ibov": 0.4386512813767289,
        "market_cap": 43532173312.0,
        "fonte_dados": "mt5"
      },
      "recomendacoes": {
        "acao_recomendada": "Adicionar ao universo de execução da próxima semana",
        "modo": "OPORTUNIDADE",
        "parametrizacao": "Usar ELITE_SYMBOLS (se existir) ou DEFAULT_PARAMS como base e refinar via otimização WFO.",
        "monitoramento": "Monitorar volatilidade e correlação diária; ajustar exposição se |corr| subir."
      },
      "prazos_e_recursos": {
        "prazo_estimado": "1 dia útil",
        "recursos_necessarios": [
          "MT5 conectado",
          "dados D1/M15",
          "cache Yahoo habilitado",
          "tempo de backtest"
        ]
      },
      "restricoes_dependencias": {
        "dependencias": [
          "MetaTrader5",
          "yfinance"
        ],
        "observacoes": [
          "Se o MT5 não tiver o símbolo, o script usa Yahoo como fallback.",
          "Backtest M15 com Yahoo é limitado e deve ser validado com dados do broker quando possível."
        ]
      },
      "justificativa_tecnica": "Liquidez OK (R$ 331.275; lim=R$ 318.889); Vol OK (21,95%; faixa=21,54%–57,52%); Corr OK (|corr|=0,439; lim=0,635); MCap OK (R$ 43.532.173.312; lim=R$ 6.132.724.480)"
    },
    {
      "identificacao": {
        "nome": "BBSE3",
        "codigo": "BBSE3",
        "codigo_unico": "BBSE3",
        "tipo": "AÇÃO",
        "categoria": "OPORTUNIDADE"
      },
      "ativo": {
        "category": "OPORTUNIDADE",
        "weight": 0.05,
        "ema_short": 9,
        "ema_long": 21,
        "rsi_low": 30.0,
        "rsi_high": 70.0,
        "adx_threshold": 25.0,
        "mom_min": 0.0,
        "sl_atr_multiplier": 2.5,
        "tp_mult": 5.0
      },
      "criterios_viabilidade": {
        "liquidez": {
          "metrica": "avg_fin_volume",
          "valor": 417630.247975,
          "limiar": 318888.73997500003,
          "condicao": ">= limiar",
          "ok": true
        },
        "volatilidade": {
          "metrica": "volatility_ann",
          "valor": 0.1532896315643996,
          "limiar_inferior": 0.21543975635038212,
          "limiar_superior": 0.5751659600971757,
          "condicao": "entre [limiar_inferior, limiar_superior]",
          "ok": false
        },
        "correlacao_ibov": {
          "metrica": "abs_corr_ibov",
          "valor": 0.4270386794176928,
          "limiar": 0.6353184419932515,
          "condicao": "<= limiar",
          "ok": true
        },
        "market_cap": {
          "metrica": "market_cap",
          "valor": 68175462400.0,
          "limiar": 6132724479.999999,
          "condicao": ">= limiar (quando limiar > 0)",
          "ok": true
        },
        "regra_final": {
          "blue_chip": false,
          "criteria_passed": 3,
          "liquidez_obrigatoria": true,
          "min_criterios": 3
        }
      },
      "resultados": {
        "rank_total": 33,
        "tier": "C",
        "score_total": 0.4845454545454545,
        "liquidez_avg_fin_volume": 417630.247975,
        "volatilidade_anualizada": 0.1532896315643996,
        "correlacao_ibov": 0.4270386794176928,
        "market_cap": 68175462400.0,
        "fonte_dados": "mt5"
      },
      "recomendacoes": {
        "acao_recomendada": "Adicionar ao universo de execução da próxima semana",
        "modo": "OPORTUNIDADE",
        "parametrizacao": "Usar ELITE_SYMBOLS (se existir) ou DEFAULT_PARAMS como base e refinar via otimização WFO.",
        "monitoramento": "Monitorar volatilidade e correlação diária; ajustar exposição se |corr| subir."
      },
      "prazos_e_recursos": {
        "prazo_estimado": "1 dia útil",
        "recursos_necessarios": [
          "MT5 conectado",
          "dados D1/M15",
          "cache Yahoo habilitado",
          "tempo de backtest"
        ]
      },
      "restricoes_dependencias": {
        "dependencias": [
          "MetaTrader5",
          "yfinance"
        ],
        "observacoes": [
          "Se o MT5 não tiver o símbolo, o script usa Yahoo como fallback.",
          "Backtest M15 com Yahoo é limitado e deve ser validado com dados do broker quando possível."
        ]
      },
      "justificativa_tecnica": "Liquidez OK (R$ 417.630; lim=R$ 318.889); Vol NOK (15,33%; faixa=21,54%–57,52%); Corr OK (|corr|=0,427; lim=0,635); MCap OK (R$ 68.175.462.400; lim=R$ 6.132.724.480)"
    },
    {
      "identificacao": {
        "nome": "VIVT3",
        "codigo": "VIVT3",
        "codigo_unico": "VIVT3",
        "tipo": "AÇÃO",
        "categoria": "OPORTUNIDADE"
      },
      "ativo": {
        "category": "OPORTUNIDADE",
        "weight": 0.121,
        "ema_short": 13,
        "ema_long": 70,
        "rsi_low": 28,
        "rsi_high": 72,
        "adx_threshold": 29,
        "mom_min": 0.0,
        "sl_atr_multiplier": 3.7,
        "tp_mult": 3.0
      },
      "criterios_viabilidade": {
        "liquidez": {
          "metrica": "avg_fin_volume",
          "valor": 352506.52770000004,
          "limiar": 318888.73997500003,
          "condicao": ">= limiar",
          "ok": true
        },
        "volatilidade": {
          "metrica": "volatility_ann",
          "valor": 0.21559613920074364,
          "limiar_inferior": 0.21543975635038212,
          "limiar_superior": 0.5751659600971757,
          "condicao": "entre [limiar_inferior, limiar_superior]",
          "ok": true
        },
        "correlacao_ibov": {
          "metrica": "abs_corr_ibov",
          "valor": 0.5285535671410724,
          "limiar": 0.6353184419932515,
          "condicao": "<= limiar",
          "ok": true
        },
        "market_cap": {
          "metrica": "market_cap",
          "valor": 103761338368.0,
          "limiar": 6132724479.999999,
          "condicao": ">= limiar (quando limiar > 0)",
          "ok": true
        },
        "regra_final": {
          "blue_chip": false,
          "criteria_passed": 4,
          "liquidez_obrigatoria": true,
          "min_criterios": 3
        }
      },
      "resultados": {
        "rank_total": 34,
        "tier": "C",
        "score_total": 0.4790909090909091,
        "liquidez_avg_fin_volume": 352506.52770000004,
        "volatilidade_anualizada": 0.21559613920074364,
        "correlacao_ibov": 0.5285535671410724,
        "market_cap": 103761338368.0,
        "fonte_dados": "mt5"
      },
      "recomendacoes": {
        "acao_recomendada": "Adicionar ao universo de execução da próxima semana",
        "modo": "OPORTUNIDADE",
        "parametrizacao": "Usar ELITE_SYMBOLS (se existir) ou DEFAULT_PARAMS como base e refinar via otimização WFO.",
        "monitoramento": "Monitorar volatilidade e correlação diária; ajustar exposição se |corr| subir."
      },
      "prazos_e_recursos": {
        "prazo_estimado": "1 dia útil",
        "recursos_necessarios": [
          "MT5 conectado",
          "dados D1/M15",
          "cache Yahoo habilitado",
          "tempo de backtest"
        ]
      },
      "restricoes_dependencias": {
        "dependencias": [
          "MetaTrader5",
          "yfinance"
        ],
        "observacoes": [
          "Se o MT5 não tiver o símbolo, o script usa Yahoo como fallback.",
          "Backtest M15 com Yahoo é limitado e deve ser validado com dados do broker quando possível."
        ]
      },
      "justificativa_tecnica": "Liquidez OK (R$ 352.507; lim=R$ 318.889); Vol OK (21,56%; faixa=21,54%–57,52%); Corr OK (|corr|=0,529; lim=0,635); MCap OK (R$ 103.761.338.368; lim=R$ 6.132.724.480)"
    },
    {
      "identificacao": {
        "nome": "TIMS3",
        "codigo": "TIMS3",
        "codigo_unico": "TIMS3",
        "tipo": "AÇÃO",
        "categoria": "OPORTUNIDADE"
      },
      "ativo": {
        "category": "OPORTUNIDADE",
        "weight": 0.05,
        "ema_short": 9,
        "ema_long": 21,
        "rsi_low": 30.0,
        "rsi_high": 70.0,
        "adx_threshold": 25.0,
        "mom_min": 0.0,
        "sl_atr_multiplier": 2.5,
        "tp_mult": 5.0
      },
      "criterios_viabilidade": {
        "liquidez": {
          "metrica": "avg_fin_volume",
          "valor": 349537.802,
          "limiar": 318888.73997500003,
          "condicao": ">= limiar",
          "ok": true
        },
        "volatilidade": {
          "metrica": "volatility_ann",
          "valor": 0.20838919569682463,
          "limiar_inferior": 0.21543975635038212,
          "limiar_superior": 0.5751659600971757,
          "condicao": "entre [limiar_inferior, limiar_superior]",
          "ok": false
        },
        "correlacao_ibov": {
          "metrica": "abs_corr_ibov",
          "valor": 0.41481586592251674,
          "limiar": 0.6353184419932515,
          "condicao": "<= limiar",
          "ok": true
        },
        "market_cap": {
          "metrica": "market_cap",
          "valor": 54652731392.0,
          "limiar": 6132724479.999999,
          "condicao": ">= limiar (quando limiar > 0)",
          "ok": true
        },
        "regra_final": {
          "blue_chip": false,
          "criteria_passed": 3,
          "liquidez_obrigatoria": true,
          "min_criterios": 3
        }
      },
      "resultados": {
        "rank_total": 35,
        "tier": "C",
        "score_total": 0.4727272727272727,
        "liquidez_avg_fin_volume": 349537.802,
        "volatilidade_anualizada": 0.20838919569682463,
        "correlacao_ibov": 0.41481586592251674,
        "market_cap": 54652731392.0,
        "fonte_dados": "mt5"
      },
      "recomendacoes": {
        "acao_recomendada": "Adicionar ao universo de execução da próxima semana",
        "modo": "OPORTUNIDADE",
        "parametrizacao": "Usar ELITE_SYMBOLS (se existir) ou DEFAULT_PARAMS como base e refinar via otimização WFO.",
        "monitoramento": "Monitorar volatilidade e correlação diária; ajustar exposição se |corr| subir."
      },
      "prazos_e_recursos": {
        "prazo_estimado": "1 dia útil",
        "recursos_necessarios": [
          "MT5 conectado",
          "dados D1/M15",
          "cache Yahoo habilitado",
          "tempo de backtest"
        ]
      },
      "restricoes_dependencias": {
        "dependencias": [
          "MetaTrader5",
          "yfinance"
        ],
        "observacoes": [
          "Se o MT5 não tiver o símbolo, o script usa Yahoo como fallback.",
          "Backtest M15 com Yahoo é limitado e deve ser validado com dados do broker quando possível."
        ]
      },
      "justificativa_tecnica": "Liquidez OK (R$ 349.538; lim=R$ 318.889); Vol NOK (20,84%; faixa=21,54%–57,52%); Corr OK (|corr|=0,415; lim=0,635); MCap OK (R$ 54.652.731.392; lim=R$ 6.132.724.480)"
    },
    {
      "identificacao": {
        "nome": "B3SA3",
        "codigo": "B3SA3",
        "codigo_unico": "B3SA3",
        "tipo": "AÇÃO",
        "categoria": "BLUE CHIP"
      },
      "ativo": {
        "category": "BLUE CHIP",
        "weight": 0.1063,
        "ema_short": 21,
        "ema_long": 90,
        "rsi_low": 34,
        "rsi_high": 60,
        "adx_threshold": 28,
        "mom_min": 0.0,
        "sl_atr_multiplier": 3.1,
        "tp_mult": 3.0
      },
      "criterios_viabilidade": {
        "liquidez": {
          "metrica": "avg_fin_volume",
          "valor": 460474.49455,
          "limiar": 318888.73997500003,
          "condicao": ">= limiar",
          "ok": true
        },
        "volatilidade": {
          "metrica": "volatility_ann",
          "valor": 0.29726555168045443,
          "limiar_inferior": 0.21543975635038212,
          "limiar_superior": 0.5751659600971757,
          "condicao": "entre [limiar_inferior, limiar_superior]",
          "ok": true
        },
        "correlacao_ibov": {
          "metrica": "abs_corr_ibov",
          "valor": 0.7661307921254287,
          "limiar": 0.6353184419932515,
          "condicao": "<= limiar",
          "ok": true
        },
        "market_cap": {
          "metrica": "market_cap",
          "valor": 76726714368.0,
          "limiar": 6132724479.999999,
          "condicao": ">= limiar (quando limiar > 0)",
          "ok": true
        },
        "regra_final": {
          "blue_chip": true,
          "criteria_passed": 4,
          "liquidez_obrigatoria": true,
          "min_criterios": 3
        }
      },
      "resultados": {
        "rank_total": 37,
        "tier": "C",
        "score_total": 0.4509090909090909,
        "liquidez_avg_fin_volume": 460474.49455,
        "volatilidade_anualizada": 0.29726555168045443,
        "correlacao_ibov": 0.7661307921254287,
        "market_cap": 76726714368.0,
        "fonte_dados": "mt5"
      },
      "recomendacoes": {
        "acao_recomendada": "Adicionar ao universo de execução da próxima semana",
        "modo": "BLUE CHIP",
        "parametrizacao": "Usar ELITE_SYMBOLS (se existir) ou DEFAULT_PARAMS como base e refinar via otimização WFO.",
        "monitoramento": "Monitorar volatilidade e correlação diária; ajustar exposição se |corr| subir."
      },
      "prazos_e_recursos": {
        "prazo_estimado": "1 dia útil",
        "recursos_necessarios": [
          "MT5 conectado",
          "dados D1/M15",
          "cache Yahoo habilitado",
          "tempo de backtest"
        ]
      },
      "restricoes_dependencias": {
        "dependencias": [
          "MetaTrader5",
          "yfinance"
        ],
        "observacoes": [
          "Se o MT5 não tiver o símbolo, o script usa Yahoo como fallback.",
          "Backtest M15 com Yahoo é limitado e deve ser validado com dados do broker quando possível."
        ]
      },
      "justificativa_tecnica": "Blue chip: marcada como viável por definição (robustez + liquidez estrutural)."
    },
    {
      "identificacao": {
        "nome": "TOTS3",
        "codigo": "TOTS3",
        "codigo_unico": "TOTS3",
        "tipo": "AÇÃO",
        "categoria": "OPORTUNIDADE"
      },
      "ativo": {
        "category": "OPORTUNIDADE",
        "weight": 0.05,
        "ema_short": 9,
        "ema_long": 21,
        "rsi_low": 30.0,
        "rsi_high": 70.0,
        "adx_threshold": 25.0,
        "mom_min": 0.0,
        "sl_atr_multiplier": 2.5,
        "tp_mult": 5.0
      },
      "criterios_viabilidade": {
        "liquidez": {
          "metrica": "avg_fin_volume",
          "valor": 478308.99417500006,
          "limiar": 318888.73997500003,
          "condicao": ">= limiar",
          "ok": true
        },
        "volatilidade": {
          "metrica": "volatility_ann",
          "valor": 0.2720103097026729,
          "limiar_inferior": 0.21543975635038212,
          "limiar_superior": 0.5751659600971757,
          "condicao": "entre [limiar_inferior, limiar_superior]",
          "ok": true
        },
        "correlacao_ibov": {
          "metrica": "abs_corr_ibov",
          "valor": 0.5015803615682356,
          "limiar": 0.6353184419932515,
          "condicao": "<= limiar",
          "ok": true
        },
        "market_cap": {
          "metrica": "market_cap",
          "valor": 25321539584.0,
          "limiar": 6132724479.999999,
          "condicao": ">= limiar (quando limiar > 0)",
          "ok": true
        },
        "regra_final": {
          "blue_chip": false,
          "criteria_passed": 4,
          "liquidez_obrigatoria": true,
          "min_criterios": 3
        }
      },
      "resultados": {
        "rank_total": 38,
        "tier": "C",
        "score_total": 0.4427272727272727,
        "liquidez_avg_fin_volume": 478308.99417500006,
        "volatilidade_anualizada": 0.2720103097026729,
        "correlacao_ibov": 0.5015803615682356,
        "market_cap": 25321539584.0,
        "fonte_dados": "mt5"
      },
      "recomendacoes": {
        "acao_recomendada": "Adicionar ao universo de execução da próxima semana",
        "modo": "OPORTUNIDADE",
        "parametrizacao": "Usar ELITE_SYMBOLS (se existir) ou DEFAULT_PARAMS como base e refinar via otimização WFO.",
        "monitoramento": "Monitorar volatilidade e correlação diária; ajustar exposição se |corr| subir."
      },
      "prazos_e_recursos": {
        "prazo_estimado": "1 dia útil",
        "recursos_necessarios": [
          "MT5 conectado",
          "dados D1/M15",
          "cache Yahoo habilitado",
          "tempo de backtest"
        ]
      },
      "restricoes_dependencias": {
        "dependencias": [
          "MetaTrader5",
          "yfinance"
        ],
        "observacoes": [
          "Se o MT5 não tiver o símbolo, o script usa Yahoo como fallback.",
          "Backtest M15 com Yahoo é limitado e deve ser validado com dados do broker quando possível."
        ]
      },
      "justificativa_tecnica": "Liquidez OK (R$ 478.309; lim=R$ 318.889); Vol OK (27,20%; faixa=21,54%–57,52%); Corr OK (|corr|=0,502; lim=0,635); MCap OK (R$ 25.321.539.584; lim=R$ 6.132.724.480)"
    },
    {
      "identificacao": {
        "nome": "ITUB4",
        "codigo": "ITUB4",
        "codigo_unico": "ITUB4",
        "tipo": "AÇÃO",
        "categoria": "BLUE CHIP"
      },
      "ativo": {
        "category": "BLUE CHIP",
        "weight": 0.0376,
        "ema_short": 25,
        "ema_long": 66,
        "rsi_low": 42,
        "rsi_high": 63,
        "adx_threshold": 23,
        "mom_min": 0.0,
        "sl_atr_multiplier": 4.5,
        "tp_mult": 3.0
      },
      "criterios_viabilidade": {
        "liquidez": {
          "metrica": "avg_fin_volume",
          "valor": 1046252.3441750001,
          "limiar": 318888.73997500003,
          "condicao": ">= limiar",
          "ok": true
        },
        "volatilidade": {
          "metrica": "volatility_ann",
          "valor": 0.18998941259086452,
          "limiar_inferior": 0.21543975635038212,
          "limiar_superior": 0.5751659600971757,
          "condicao": "entre [limiar_inferior, limiar_superior]",
          "ok": true
        },
        "correlacao_ibov": {
          "metrica": "abs_corr_ibov",
          "valor": 0.8562278410740507,
          "limiar": 0.6353184419932515,
          "condicao": "<= limiar",
          "ok": true
        },
        "market_cap": {
          "metrica": "market_cap",
          "valor": 436760641536.0,
          "limiar": 6132724479.999999,
          "condicao": ">= limiar (quando limiar > 0)",
          "ok": true
        },
        "regra_final": {
          "blue_chip": true,
          "criteria_passed": 4,
          "liquidez_obrigatoria": true,
          "min_criterios": 3
        }
      },
      "resultados": {
        "rank_total": 39,
        "tier": "D",
        "score_total": 0.4409090909090909,
        "liquidez_avg_fin_volume": 1046252.3441750001,
        "volatilidade_anualizada": 0.18998941259086452,
        "correlacao_ibov": 0.8562278410740507,
        "market_cap": 436760641536.0,
        "fonte_dados": "mt5"
      },
      "recomendacoes": {
        "acao_recomendada": "Adicionar ao universo de execução da próxima semana",
        "modo": "BLUE CHIP",
        "parametrizacao": "Usar ELITE_SYMBOLS (se existir) ou DEFAULT_PARAMS como base e refinar via otimização WFO.",
        "monitoramento": "Monitorar volatilidade e correlação diária; ajustar exposição se |corr| subir."
      },
      "prazos_e_recursos": {
        "prazo_estimado": "1 dia útil",
        "recursos_necessarios": [
          "MT5 conectado",
          "dados D1/M15",
          "cache Yahoo habilitado",
          "tempo de backtest"
        ]
      },
      "restricoes_dependencias": {
        "dependencias": [
          "MetaTrader5",
          "yfinance"
        ],
        "observacoes": [
          "Se o MT5 não tiver o símbolo, o script usa Yahoo como fallback.",
          "Backtest M15 com Yahoo é limitado e deve ser validado com dados do broker quando possível."
        ]
      },
      "justificativa_tecnica": "Blue chip: marcada como viável por definição (robustez + liquidez estrutural)."
    },
    {
      "identificacao": {
        "nome": "BBDC4",
        "codigo": "BBDC4",
        "codigo_unico": "BBDC4",
        "tipo": "AÇÃO",
        "categoria": "BLUE CHIP"
      },
      "ativo": {
        "category": "BLUE CHIP",
        "weight": 0.1106,
        "ema_short": 14,
        "ema_long": 58,
        "rsi_low": 38,
        "rsi_high": 61,
        "adx_threshold": 12,
        "mom_min": 0.0,
        "sl_atr_multiplier": 4.300000000000001,
        "tp_mult": 3.0
      },
      "criterios_viabilidade": {
        "liquidez": {
          "metrica": "avg_fin_volume",
          "valor": 514458.65184999985,
          "limiar": 318888.73997500003,
          "condicao": ">= limiar",
          "ok": true
        },
        "volatilidade": {
          "metrica": "volatility_ann",
          "valor": 0.23302512616331544,
          "limiar_inferior": 0.21543975635038212,
          "limiar_superior": 0.5751659600971757,
          "condicao": "entre [limiar_inferior, limiar_superior]",
          "ok": true
        },
        "correlacao_ibov": {
          "metrica": "abs_corr_ibov",
          "valor": 0.7850135103673879,
          "limiar": 0.6353184419932515,
          "condicao": "<= limiar",
          "ok": true
        },
        "market_cap": {
          "metrica": "market_cap",
          "valor": 200011284480.0,
          "limiar": 6132724479.999999,
          "condicao": ">= limiar (quando limiar > 0)",
          "ok": true
        },
        "regra_final": {
          "blue_chip": true,
          "criteria_passed": 4,
          "liquidez_obrigatoria": true,
          "min_criterios": 3
        }
      },
      "resultados": {
        "rank_total": 40,
        "tier": "D",
        "score_total": 0.4354545454545454,
        "liquidez_avg_fin_volume": 514458.65184999985,
        "volatilidade_anualizada": 0.23302512616331544,
        "correlacao_ibov": 0.7850135103673879,
        "market_cap": 200011284480.0,
        "fonte_dados": "mt5"
      },
      "recomendacoes": {
        "acao_recomendada": "Adicionar ao universo de execução da próxima semana",
        "modo": "BLUE CHIP",
        "parametrizacao": "Usar ELITE_SYMBOLS (se existir) ou DEFAULT_PARAMS como base e refinar via otimização WFO.",
        "monitoramento": "Monitorar volatilidade e correlação diária; ajustar exposição se |corr| subir."
      },
      "prazos_e_recursos": {
        "prazo_estimado": "1 dia útil",
        "recursos_necessarios": [
          "MT5 conectado",
          "dados D1/M15",
          "cache Yahoo habilitado",
          "tempo de backtest"
        ]
      },
      "restricoes_dependencias": {
        "dependencias": [
          "MetaTrader5",
          "yfinance"
        ],
        "observacoes": [
          "Se o MT5 não tiver o símbolo, o script usa Yahoo como fallback.",
          "Backtest M15 com Yahoo é limitado e deve ser validado com dados do broker quando possível."
        ]
      },
      "justificativa_tecnica": "Blue chip: marcada como viável por definição (robustez + liquidez estrutural)."
    },
    {
      "identificacao": {
        "nome": "ABEV3",
        "codigo": "ABEV3",
        "codigo_unico": "ABEV3",
        "tipo": "AÇÃO",
        "categoria": "BLUE CHIP"
      },
      "ativo": {
        "category": "BLUE CHIP",
        "weight": 0.0,
        "ema_short": 12,
        "ema_long": 97,
        "rsi_low": 37,
        "rsi_high": 73,
        "adx_threshold": 13,
        "mom_min": 0.0,
        "sl_atr_multiplier": 1.9,
        "tp_mult": 3.0
      },
      "criterios_viabilidade": {
        "liquidez": {
          "metrica": "avg_fin_volume",
          "valor": 338391.43375,
          "limiar": 318888.73997500003,
          "condicao": ">= limiar",
          "ok": true
        },
        "volatilidade": {
          "metrica": "volatility_ann",
          "valor": 0.20256692728683867,
          "limiar_inferior": 0.21543975635038212,
          "limiar_superior": 0.5751659600971757,
          "condicao": "entre [limiar_inferior, limiar_superior]",
          "ok": true
        },
        "correlacao_ibov": {
          "metrica": "abs_corr_ibov",
          "valor": 0.4256381642173374,
          "limiar": 0.6353184419932515,
          "condicao": "<= limiar",
          "ok": true
        },
        "market_cap": {
          "metrica": "market_cap",
          "valor": 220193013760.0,
          "limiar": 6132724479.999999,
          "condicao": ">= limiar (quando limiar > 0)",
          "ok": true
        },
        "regra_final": {
          "blue_chip": true,
          "criteria_passed": 4,
          "liquidez_obrigatoria": true,
          "min_criterios": 3
        }
      },
      "resultados": {
        "rank_total": 41,
        "tier": "D",
        "score_total": 0.42727272727272725,
        "liquidez_avg_fin_volume": 338391.43375,
        "volatilidade_anualizada": 0.20256692728683867,
        "correlacao_ibov": 0.4256381642173374,
        "market_cap": 220193013760.0,
        "fonte_dados": "mt5"
      },
      "recomendacoes": {
        "acao_recomendada": "Adicionar ao universo de execução da próxima semana",
        "modo": "BLUE CHIP",
        "parametrizacao": "Usar ELITE_SYMBOLS (se existir) ou DEFAULT_PARAMS como base e refinar via otimização WFO.",
        "monitoramento": "Monitorar volatilidade e correlação diária; ajustar exposição se |corr| subir."
      },
      "prazos_e_recursos": {
        "prazo_estimado": "1 dia útil",
        "recursos_necessarios": [
          "MT5 conectado",
          "dados D1/M15",
          "cache Yahoo habilitado",
          "tempo de backtest"
        ]
      },
      "restricoes_dependencias": {
        "dependencias": [
          "MetaTrader5",
          "yfinance"
        ],
        "observacoes": [
          "Se o MT5 não tiver o símbolo, o script usa Yahoo como fallback.",
          "Backtest M15 com Yahoo é limitado e deve ser validado com dados do broker quando possível."
        ]
      },
      "justificativa_tecnica": "Blue chip: marcada como viável por definição (robustez + liquidez estrutural)."
    },
    {
      "identificacao": {
        "nome": "RENT3",
        "codigo": "RENT3",
        "codigo_unico": "RENT3",
        "tipo": "AÇÃO",
        "categoria": "OPORTUNIDADE"
      },
      "ativo": {
        "category": "OPORTUNIDADE",
        "weight": 0.05,
        "ema_short": 9,
        "ema_long": 21,
        "rsi_low": 30.0,
        "rsi_high": 70.0,
        "adx_threshold": 25.0,
        "mom_min": 0.0,
        "sl_atr_multiplier": 2.5,
        "tp_mult": 5.0
      },
      "criterios_viabilidade": {
        "liquidez": {
          "metrica": "avg_fin_volume",
          "valor": 1064569.070275,
          "limiar": 318888.73997500003,
          "condicao": ">= limiar",
          "ok": true
        },
        "volatilidade": {
          "metrica": "volatility_ann",
          "valor": 0.3185563477870126,
          "limiar_inferior": 0.21543975635038212,
          "limiar_superior": 0.5751659600971757,
          "condicao": "entre [limiar_inferior, limiar_superior]",
          "ok": true
        },
        "correlacao_ibov": {
          "metrica": "abs_corr_ibov",
          "valor": 0.7917082872640059,
          "limiar": 0.6353184419932515,
          "condicao": "<= limiar",
          "ok": false
        },
        "market_cap": {
          "metrica": "market_cap",
          "valor": 42917515264.0,
          "limiar": 6132724479.999999,
          "condicao": ">= limiar (quando limiar > 0)",
          "ok": true
        },
        "regra_final": {
          "blue_chip": false,
          "criteria_passed": 3,
          "liquidez_obrigatoria": true,
          "min_criterios": 3
        }
      },
      "resultados": {
        "rank_total": 43,
        "tier": "D",
        "score_total": 0.4163636363636364,
        "liquidez_avg_fin_volume": 1064569.070275,
        "volatilidade_anualizada": 0.3185563477870126,
        "correlacao_ibov": 0.7917082872640059,
        "market_cap": 42917515264.0,
        "fonte_dados": "mt5"
      },
      "recomendacoes": {
        "acao_recomendada": "Adicionar ao universo de execução da próxima semana",
        "modo": "OPORTUNIDADE",
        "parametrizacao": "Usar ELITE_SYMBOLS (se existir) ou DEFAULT_PARAMS como base e refinar via otimização WFO.",
        "monitoramento": "Monitorar volatilidade e correlação diária; ajustar exposição se |corr| subir."
      },
      "prazos_e_recursos": {
        "prazo_estimado": "1 dia útil",
        "recursos_necessarios": [
          "MT5 conectado",
          "dados D1/M15",
          "cache Yahoo habilitado",
          "tempo de backtest"
        ]
      },
      "restricoes_dependencias": {
        "dependencias": [
          "MetaTrader5",
          "yfinance"
        ],
        "observacoes": [
          "Se o MT5 não tiver o símbolo, o script usa Yahoo como fallback.",
          "Backtest M15 com Yahoo é limitado e deve ser validado com dados do broker quando possível."
        ]
      },
      "justificativa_tecnica": "Liquidez OK (R$ 1.064.569; lim=R$ 318.889); Vol OK (31,86%; faixa=21,54%–57,52%); Corr NOK (|corr|=0,792; lim=0,635); MCap OK (R$ 42.917.515.264; lim=R$ 6.132.724.480)"
    },
    {
      "identificacao": {
        "nome": "BBAS3",
        "codigo": "BBAS3",
        "codigo_unico": "BBAS3",
        "tipo": "AÇÃO",
        "categoria": "BLUE CHIP"
      },
      "ativo": {
        "category": "BLUE CHIP",
        "weight": 0.0,
        "ema_short": 12,
        "ema_long": 97,
        "rsi_low": 37,
        "rsi_high": 73,
        "adx_threshold": 13,
        "mom_min": 0.0,
        "sl_atr_multiplier": 1.9,
        "tp_mult": 3.0
      },
      "criterios_viabilidade": {
        "liquidez": {
          "metrica": "avg_fin_volume",
          "valor": 489877.67972499994,
          "limiar": 318888.73997500003,
          "condicao": ">= limiar",
          "ok": true
        },
        "volatilidade": {
          "metrica": "volatility_ann",
          "valor": 0.2858455965155945,
          "limiar_inferior": 0.21543975635038212,
          "limiar_superior": 0.5751659600971757,
          "condicao": "entre [limiar_inferior, limiar_superior]",
          "ok": true
        },
        "correlacao_ibov": {
          "metrica": "abs_corr_ibov",
          "valor": 0.6455076144638412,
          "limiar": 0.6353184419932515,
          "condicao": "<= limiar",
          "ok": true
        },
        "market_cap": {
          "metrica": "market_cap",
          "valor": 121988308992.0,
          "limiar": 6132724479.999999,
          "condicao": ">= limiar (quando limiar > 0)",
          "ok": true
        },
        "regra_final": {
          "blue_chip": true,
          "criteria_passed": 4,
          "liquidez_obrigatoria": true,
          "min_criterios": 3
        }
      },
      "resultados": {
        "rank_total": 44,
        "tier": "D",
        "score_total": 0.40272727272727277,
        "liquidez_avg_fin_volume": 489877.67972499994,
        "volatilidade_anualizada": 0.2858455965155945,
        "correlacao_ibov": 0.6455076144638412,
        "market_cap": 121988308992.0,
        "fonte_dados": "mt5"
      },
      "recomendacoes": {
        "acao_recomendada": "Adicionar ao universo de execução da próxima semana",
        "modo": "BLUE CHIP",
        "parametrizacao": "Usar ELITE_SYMBOLS (se existir) ou DEFAULT_PARAMS como base e refinar via otimização WFO.",
        "monitoramento": "Monitorar volatilidade e correlação diária; ajustar exposição se |corr| subir."
      },
      "prazos_e_recursos": {
        "prazo_estimado": "1 dia útil",
        "recursos_necessarios": [
          "MT5 conectado",
          "dados D1/M15",
          "cache Yahoo habilitado",
          "tempo de backtest"
        ]
      },
      "restricoes_dependencias": {
        "dependencias": [
          "MetaTrader5",
          "yfinance"
        ],
        "observacoes": [
          "Se o MT5 não tiver o símbolo, o script usa Yahoo como fallback.",
          "Backtest M15 com Yahoo é limitado e deve ser validado com dados do broker quando possível."
        ]
      },
      "justificativa_tecnica": "Blue chip: marcada como viável por definição (robustez + liquidez estrutural)."
    },
    {
      "identificacao": {
        "nome": "RDOR3",
        "codigo": "RDOR3",
        "codigo_unico": "RDOR3",
        "tipo": "AÇÃO",
        "categoria": "OPORTUNIDADE"
      },
      "ativo": {
        "category": "OPORTUNIDADE",
        "weight": 0.05,
        "ema_short": 9,
        "ema_long": 21,
        "rsi_low": 30.0,
        "rsi_high": 70.0,
        "adx_threshold": 25.0,
        "mom_min": 0.0,
        "sl_atr_multiplier": 2.5,
        "tp_mult": 5.0
      },
      "criterios_viabilidade": {
        "liquidez": {
          "metrica": "avg_fin_volume",
          "valor": 843224.6680500002,
          "limiar": 318888.73997500003,
          "condicao": ">= limiar",
          "ok": true
        },
        "volatilidade": {
          "metrica": "volatility_ann",
          "valor": 0.27950854271455355,
          "limiar_inferior": 0.21543975635038212,
          "limiar_superior": 0.5751659600971757,
          "condicao": "entre [limiar_inferior, limiar_superior]",
          "ok": true
        },
        "correlacao_ibov": {
          "metrica": "abs_corr_ibov",
          "valor": 0.6300774811235531,
          "limiar": 0.6353184419932515,
          "condicao": "<= limiar",
          "ok": true
        },
        "market_cap": {
          "metrica": "market_cap",
          "valor": 88808005632.0,
          "limiar": 6132724479.999999,
          "condicao": ">= limiar (quando limiar > 0)",
          "ok": true
        },
        "regra_final": {
          "blue_chip": false,
          "criteria_passed": 4,
          "liquidez_obrigatoria": true,
          "min_criterios": 3
        }
      },
      "resultados": {
        "rank_total": 45,
        "tier": "D",
        "score_total": 0.3781818181818182,
        "liquidez_avg_fin_volume": 843224.6680500002,
        "volatilidade_anualizada": 0.27950854271455355,
        "correlacao_ibov": 0.6300774811235531,
        "market_cap": 88808005632.0,
        "fonte_dados": "mt5"
      },
      "recomendacoes": {
        "acao_recomendada": "Adicionar ao universo de execução da próxima semana",
        "modo": "OPORTUNIDADE",
        "parametrizacao": "Usar ELITE_SYMBOLS (se existir) ou DEFAULT_PARAMS como base e refinar via otimização WFO.",
        "monitoramento": "Monitorar volatilidade e correlação diária; ajustar exposição se |corr| subir."
      },
      "prazos_e_recursos": {
        "prazo_estimado": "1 dia útil",
        "recursos_necessarios": [
          "MT5 conectado",
          "dados D1/M15",
          "cache Yahoo habilitado",
          "tempo de backtest"
        ]
      },
      "restricoes_dependencias": {
        "dependencias": [
          "MetaTrader5",
          "yfinance"
        ],
        "observacoes": [
          "Se o MT5 não tiver o símbolo, o script usa Yahoo como fallback.",
          "Backtest M15 com Yahoo é limitado e deve ser validado com dados do broker quando possível."
        ]
      },
      "justificativa_tecnica": "Liquidez OK (R$ 843.225; lim=R$ 318.889); Vol OK (27,95%; faixa=21,54%–57,52%); Corr OK (|corr|=0,630; lim=0,635); MCap OK (R$ 88.808.005.632; lim=R$ 6.132.724.480)"
    },
    {
      "identificacao": {
        "nome": "SBSP3",
        "codigo": "SBSP3",
        "codigo_unico": "SBSP3",
        "tipo": "AÇÃO",
        "categoria": "OPORTUNIDADE"
      },
      "ativo": {
        "category": "BLUE CHIP",
        "weight": 0.0,
        "ema_short": 12,
        "ema_long": 97,
        "rsi_low": 37,
        "rsi_high": 73,
        "adx_threshold": 13,
        "mom_min": 0.0,
        "sl_atr_multiplier": 1.9,
        "tp_mult": 3.0
      },
      "criterios_viabilidade": {
        "liquidez": {
          "metrica": "avg_fin_volume",
          "valor": 2147835.186225,
          "limiar": 318888.73997500003,
          "condicao": ">= limiar",
          "ok": true
        },
        "volatilidade": {
          "metrica": "volatility_ann",
          "valor": 0.2713172944961507,
          "limiar_inferior": 0.21543975635038212,
          "limiar_superior": 0.5751659600971757,
          "condicao": "entre [limiar_inferior, limiar_superior]",
          "ok": true
        },
        "correlacao_ibov": {
          "metrica": "abs_corr_ibov",
          "valor": 0.6504687914560192,
          "limiar": 0.6353184419932515,
          "condicao": "<= limiar",
          "ok": false
        },
        "market_cap": {
          "metrica": "market_cap",
          "valor": 86589136896.0,
          "limiar": 6132724479.999999,
          "condicao": ">= limiar (quando limiar > 0)",
          "ok": true
        },
        "regra_final": {
          "blue_chip": false,
          "criteria_passed": 3,
          "liquidez_obrigatoria": true,
          "min_criterios": 3
        }
      },
      "resultados": {
        "rank_total": 46,
        "tier": "D",
        "score_total": 0.3618181818181818,
        "liquidez_avg_fin_volume": 2147835.186225,
        "volatilidade_anualizada": 0.2713172944961507,
        "correlacao_ibov": 0.6504687914560192,
        "market_cap": 86589136896.0,
        "fonte_dados": "mt5"
      },
      "recomendacoes": {
        "acao_recomendada": "Adicionar ao universo de execução da próxima semana",
        "modo": "OPORTUNIDADE",
        "parametrizacao": "Usar ELITE_SYMBOLS (se existir) ou DEFAULT_PARAMS como base e refinar via otimização WFO.",
        "monitoramento": "Monitorar volatilidade e correlação diária; ajustar exposição se |corr| subir."
      },
      "prazos_e_recursos": {
        "prazo_estimado": "1 dia útil",
        "recursos_necessarios": [
          "MT5 conectado",
          "dados D1/M15",
          "cache Yahoo habilitado",
          "tempo de backtest"
        ]
      },
      "restricoes_dependencias": {
        "dependencias": [
          "MetaTrader5",
          "yfinance"
        ],
        "observacoes": [
          "Se o MT5 não tiver o símbolo, o script usa Yahoo como fallback.",
          "Backtest M15 com Yahoo é limitado e deve ser validado com dados do broker quando possível."
        ]
      },
      "justificativa_tecnica": "Liquidez OK (R$ 2.147.835; lim=R$ 318.889); Vol OK (27,13%; faixa=21,54%–57,52%); Corr NOK (|corr|=0,650; lim=0,635); MCap OK (R$ 86.589.136.896; lim=R$ 6.132.724.480)"
    },
    {
      "identificacao": {
        "nome": "PRIO3",
        "codigo": "PRIO3",
        "codigo_unico": "PRIO3",
        "tipo": "AÇÃO",
        "categoria": "OPORTUNIDADE"
      },
      "ativo": {
        "category": "BLUE CHIP",
        "weight": 0.0902,
        "ema_short": 13,
        "ema_long": 76,
        "rsi_low": 28,
        "rsi_high": 80,
        "adx_threshold": 29,
        "mom_min": 0.0,
        "sl_atr_multiplier": 4.300000000000001,
        "tp_mult": 3.0
      },
      "criterios_viabilidade": {
        "liquidez": {
          "metrica": "avg_fin_volume",
          "valor": 1045277.529075,
          "limiar": 318888.73997500003,
          "condicao": ">= limiar",
          "ok": true
        },
        "volatilidade": {
          "metrica": "volatility_ann",
          "valor": 0.23901108867789483,
          "limiar_inferior": 0.21543975635038212,
          "limiar_superior": 0.5751659600971757,
          "condicao": "entre [limiar_inferior, limiar_superior]",
          "ok": true
        },
        "correlacao_ibov": {
          "metrica": "abs_corr_ibov",
          "valor": 0.3028296082883377,
          "limiar": 0.6353184419932515,
          "condicao": "<= limiar",
          "ok": true
        },
        "market_cap": {
          "metrica": "market_cap",
          "valor": 35904335872.0,
          "limiar": 6132724479.999999,
          "condicao": ">= limiar (quando limiar > 0)",
          "ok": true
        },
        "regra_final": {
          "blue_chip": false,
          "criteria_passed": 4,
          "liquidez_obrigatoria": true,
          "min_criterios": 3
        }
      },
      "resultados": {
        "rank_total": 47,
        "tier": "D",
        "score_total": 0.34818181818181815,
        "liquidez_avg_fin_volume": 1045277.529075,
        "volatilidade_anualizada": 0.23901108867789483,
        "correlacao_ibov": 0.3028296082883377,
        "market_cap": 35904335872.0,
        "fonte_dados": "mt5"
      },
      "recomendacoes": {
        "acao_recomendada": "Adicionar ao universo de execução da próxima semana",
        "modo": "OPORTUNIDADE",
        "parametrizacao": "Usar ELITE_SYMBOLS (se existir) ou DEFAULT_PARAMS como base e refinar via otimização WFO.",
        "monitoramento": "Monitorar volatilidade e correlação diária; ajustar exposição se |corr| subir."
      },
      "prazos_e_recursos": {
        "prazo_estimado": "1 dia útil",
        "recursos_necessarios": [
          "MT5 conectado",
          "dados D1/M15",
          "cache Yahoo habilitado",
          "tempo de backtest"
        ]
      },
      "restricoes_dependencias": {
        "dependencias": [
          "MetaTrader5",
          "yfinance"
        ],
        "observacoes": [
          "Se o MT5 não tiver o símbolo, o script usa Yahoo como fallback.",
          "Backtest M15 com Yahoo é limitado e deve ser validado com dados do broker quando possível."
        ]
      },
      "justificativa_tecnica": "Liquidez OK (R$ 1.045.278; lim=R$ 318.889); Vol OK (23,90%; faixa=21,54%–57,52%); Corr OK (|corr|=0,303; lim=0,635); MCap OK (R$ 35.904.335.872; lim=R$ 6.132.724.480)"
    },
    {
      "identificacao": {
        "nome": "BPAC11",
        "codigo": "BPAC11",
        "codigo_unico": "BPAC11",
        "tipo": "UNIT/ETF",
        "categoria": "OPORTUNIDADE"
      },
      "ativo": {
        "category": "BLUE CHIP",
        "weight": 0.0112,
        "ema_short": 25,
        "ema_long": 76,
        "rsi_low": 45,
        "rsi_high": 55,
        "adx_threshold": 30,
        "mom_min": 0.0,
        "sl_atr_multiplier": 4.300000000000001,
        "tp_mult": 3.0
      },
      "criterios_viabilidade": {
        "liquidez": {
          "metrica": "avg_fin_volume",
          "valor": 1071598.297,
          "limiar": 318888.73997500003,
          "condicao": ">= limiar",
          "ok": true
        },
        "volatilidade": {
          "metrica": "volatility_ann",
          "valor": 0.3302677339933438,
          "limiar_inferior": 0.21543975635038212,
          "limiar_superior": 0.5751659600971757,
          "condicao": "entre [limiar_inferior, limiar_superior]",
          "ok": true
        },
        "correlacao_ibov": {
          "metrica": "abs_corr_ibov",
          "valor": 0.7605930579742672,
          "limiar": 0.6353184419932515,
          "condicao": "<= limiar",
          "ok": false
        },
        "market_cap": {
          "metrica": "market_cap",
          "valor": 181809987584.0,
          "limiar": 6132724479.999999,
          "condicao": ">= limiar (quando limiar > 0)",
          "ok": true
        },
        "regra_final": {
          "blue_chip": false,
          "criteria_passed": 3,
          "liquidez_obrigatoria": true,
          "min_criterios": 3
        }
      },
      "resultados": {
        "rank_total": 48,
        "tier": "D",
        "score_total": 0.3318181818181818,
        "liquidez_avg_fin_volume": 1071598.297,
        "volatilidade_anualizada": 0.3302677339933438,
        "correlacao_ibov": 0.7605930579742672,
        "market_cap": 181809987584.0,
        "fonte_dados": "mt5"
      },
      "recomendacoes": {
        "acao_recomendada": "Adicionar ao universo de execução da próxima semana",
        "modo": "OPORTUNIDADE",
        "parametrizacao": "Usar ELITE_SYMBOLS (se existir) ou DEFAULT_PARAMS como base e refinar via otimização WFO.",
        "monitoramento": "Monitorar volatilidade e correlação diária; ajustar exposição se |corr| subir."
      },
      "prazos_e_recursos": {
        "prazo_estimado": "1 dia útil",
        "recursos_necessarios": [
          "MT5 conectado",
          "dados D1/M15",
          "cache Yahoo habilitado",
          "tempo de backtest"
        ]
      },
      "restricoes_dependencias": {
        "dependencias": [
          "MetaTrader5",
          "yfinance"
        ],
        "observacoes": [
          "Se o MT5 não tiver o símbolo, o script usa Yahoo como fallback.",
          "Backtest M15 com Yahoo é limitado e deve ser validado com dados do broker quando possível."
        ]
      },
      "justificativa_tecnica": "Liquidez OK (R$ 1.071.598; lim=R$ 318.889); Vol OK (33,03%; faixa=21,54%–57,52%); Corr NOK (|corr|=0,761; lim=0,635); MCap OK (R$ 181.809.987.584; lim=R$ 6.132.724.480)"
    },
    {
      "identificacao": {
        "nome": "RADL3",
        "codigo": "RADL3",
        "codigo_unico": "RADL3",
        "tipo": "AÇÃO",
        "categoria": "OPORTUNIDADE"
      },
      "ativo": {
        "category": "OPORTUNIDADE",
        "weight": 0.0316,
        "ema_short": 24,
        "ema_long": 37,
        "rsi_low": 15,
        "rsi_high": 75,
        "adx_threshold": 19,
        "mom_min": 0.0,
        "sl_atr_multiplier": 3.1,
        "tp_mult": 3.0
      },
      "criterios_viabilidade": {
        "liquidez": {
          "metrica": "avg_fin_volume",
          "valor": 476454.10327500006,
          "limiar": 318888.73997500003,
          "condicao": ">= limiar",
          "ok": true
        },
        "volatilidade": {
          "metrica": "volatility_ann",
          "valor": 0.4309464800610215,
          "limiar_inferior": 0.21543975635038212,
          "limiar_superior": 0.5751659600971757,
          "condicao": "entre [limiar_inferior, limiar_superior]",
          "ok": true
        },
        "correlacao_ibov": {
          "metrica": "abs_corr_ibov",
          "valor": 0.4349283059375325,
          "limiar": 0.6353184419932515,
          "condicao": "<= limiar",
          "ok": true
        },
        "market_cap": {
          "metrica": "market_cap",
          "valor": 44733452288.0,
          "limiar": 6132724479.999999,
          "condicao": ">= limiar (quando limiar > 0)",
          "ok": true
        },
        "regra_final": {
          "blue_chip": false,
          "criteria_passed": 4,
          "liquidez_obrigatoria": true,
          "min_criterios": 3
        }
      },
      "resultados": {
        "rank_total": 49,
        "tier": "D",
        "score_total": 0.32909090909090905,
        "liquidez_avg_fin_volume": 476454.10327500006,
        "volatilidade_anualizada": 0.4309464800610215,
        "correlacao_ibov": 0.4349283059375325,
        "market_cap": 44733452288.0,
        "fonte_dados": "mt5"
      },
      "recomendacoes": {
        "acao_recomendada": "Adicionar ao universo de execução da próxima semana",
        "modo": "OPORTUNIDADE",
        "parametrizacao": "Usar ELITE_SYMBOLS (se existir) ou DEFAULT_PARAMS como base e refinar via otimização WFO.",
        "monitoramento": "Monitorar volatilidade e correlação diária; ajustar exposição se |corr| subir."
      },
      "prazos_e_recursos": {
        "prazo_estimado": "1 dia útil",
        "recursos_necessarios": [
          "MT5 conectado",
          "dados D1/M15",
          "cache Yahoo habilitado",
          "tempo de backtest"
        ]
      },
      "restricoes_dependencias": {
        "dependencias": [
          "MetaTrader5",
          "yfinance"
        ],
        "observacoes": [
          "Se o MT5 não tiver o símbolo, o script usa Yahoo como fallback.",
          "Backtest M15 com Yahoo é limitado e deve ser validado com dados do broker quando possível."
        ]
      },
      "justificativa_tecnica": "Liquidez OK (R$ 476.454; lim=R$ 318.889); Vol OK (43,09%; faixa=21,54%–57,52%); Corr OK (|corr|=0,435; lim=0,635); MCap OK (R$ 44.733.452.288; lim=R$ 6.132.724.480)"
    },
    {
      "identificacao": {
        "nome": "SUZB3",
        "codigo": "SUZB3",
        "codigo_unico": "SUZB3",
        "tipo": "AÇÃO",
        "categoria": "BLUE CHIP"
      },
      "ativo": {
        "category": "BLUE CHIP",
        "weight": 0.05,
        "ema_short": 9,
        "ema_long": 21,
        "rsi_low": 30.0,
        "rsi_high": 70.0,
        "adx_threshold": 25.0,
        "mom_min": 0.0,
        "sl_atr_multiplier": 2.5,
        "tp_mult": 5.0
      },
      "criterios_viabilidade": {
        "liquidez": {
          "metrica": "avg_fin_volume",
          "valor": 967376.165925,
          "limiar": 318888.73997500003,
          "condicao": ">= limiar",
          "ok": true
        },
        "volatilidade": {
          "metrica": "volatility_ann",
          "valor": 0.214814224948936,
          "limiar_inferior": 0.21543975635038212,
          "limiar_superior": 0.5751659600971757,
          "condicao": "entre [limiar_inferior, limiar_superior]",
          "ok": true
        },
        "correlacao_ibov": {
          "metrica": "abs_corr_ibov",
          "valor": 0.0589587000333559,
          "limiar": 0.6353184419932515,
          "condicao": "<= limiar",
          "ok": true
        },
        "market_cap": {
          "metrica": "market_cap",
          "valor": 63772889088.0,
          "limiar": 6132724479.999999,
          "condicao": ">= limiar (quando limiar > 0)",
          "ok": true
        },
        "regra_final": {
          "blue_chip": true,
          "criteria_passed": 4,
          "liquidez_obrigatoria": true,
          "min_criterios": 3
        }
      },
      "resultados": {
        "rank_total": 50,
        "tier": "D",
        "score_total": 0.31090909090909097,
        "liquidez_avg_fin_volume": 967376.165925,
        "volatilidade_anualizada": 0.214814224948936,
        "correlacao_ibov": 0.0589587000333559,
        "market_cap": 63772889088.0,
        "fonte_dados": "mt5"
      },
      "recomendacoes": {
        "acao_recomendada": "Adicionar ao universo de execução da próxima semana",
        "modo": "BLUE CHIP",
        "parametrizacao": "Usar ELITE_SYMBOLS (se existir) ou DEFAULT_PARAMS como base e refinar via otimização WFO.",
        "monitoramento": "Monitorar volatilidade e correlação diária; ajustar exposição se |corr| subir."
      },
      "prazos_e_recursos": {
        "prazo_estimado": "1 dia útil",
        "recursos_necessarios": [
          "MT5 conectado",
          "dados D1/M15",
          "cache Yahoo habilitado",
          "tempo de backtest"
        ]
      },
      "restricoes_dependencias": {
        "dependencias": [
          "MetaTrader5",
          "yfinance"
        ],
        "observacoes": [
          "Se o MT5 não tiver o símbolo, o script usa Yahoo como fallback.",
          "Backtest M15 com Yahoo é limitado e deve ser validado com dados do broker quando possível."
        ]
      },
      "justificativa_tecnica": "Blue chip: marcada como viável por definição (robustez + liquidez estrutural)."
    },
    {
      "identificacao": {
        "nome": "PETR4",
        "codigo": "PETR4",
        "codigo_unico": "PETR4",
        "tipo": "AÇÃO",
        "categoria": "BLUE CHIP"
      },
      "ativo": {
        "category": "BLUE CHIP",
        "weight": 0.1047,
        "ema_short": 11,
        "ema_long": 56,
        "rsi_low": 20,
        "rsi_high": 72,
        "adx_threshold": 12,
        "mom_min": 0.0,
        "sl_atr_multiplier": 3.7,
        "tp_mult": 3.0
      },
      "criterios_viabilidade": {
        "liquidez": {
          "metrica": "avg_fin_volume",
          "valor": 1210223.0015,
          "limiar": 318888.73997500003,
          "condicao": ">= limiar",
          "ok": true
        },
        "volatilidade": {
          "metrica": "volatility_ann",
          "valor": 0.2010123965519548,
          "limiar_inferior": 0.21543975635038212,
          "limiar_superior": 0.5751659600971757,
          "condicao": "entre [limiar_inferior, limiar_superior]",
          "ok": true
        },
        "correlacao_ibov": {
          "metrica": "abs_corr_ibov",
          "valor": 0.4538822359303427,
          "limiar": 0.6353184419932515,
          "condicao": "<= limiar",
          "ok": true
        },
        "market_cap": {
          "metrica": "market_cap",
          "valor": 430539309056.0,
          "limiar": 6132724479.999999,
          "condicao": ">= limiar (quando limiar > 0)",
          "ok": true
        },
        "regra_final": {
          "blue_chip": true,
          "criteria_passed": 4,
          "liquidez_obrigatoria": true,
          "min_criterios": 3
        }
      },
      "resultados": {
        "rank_total": 51,
        "tier": "D",
        "score_total": 0.3063636363636364,
        "liquidez_avg_fin_volume": 1210223.0015,
        "volatilidade_anualizada": 0.2010123965519548,
        "correlacao_ibov": 0.4538822359303427,
        "market_cap": 430539309056.0,
        "fonte_dados": "mt5"
      },
      "recomendacoes": {
        "acao_recomendada": "Adicionar ao universo de execução da próxima semana",
        "modo": "BLUE CHIP",
        "parametrizacao": "Usar ELITE_SYMBOLS (se existir) ou DEFAULT_PARAMS como base e refinar via otimização WFO.",
        "monitoramento": "Monitorar volatilidade e correlação diária; ajustar exposição se |corr| subir."
      },
      "prazos_e_recursos": {
        "prazo_estimado": "1 dia útil",
        "recursos_necessarios": [
          "MT5 conectado",
          "dados D1/M15",
          "cache Yahoo habilitado",
          "tempo de backtest"
        ]
      },
      "restricoes_dependencias": {
        "dependencias": [
          "MetaTrader5",
          "yfinance"
        ],
        "observacoes": [
          "Se o MT5 não tiver o símbolo, o script usa Yahoo como fallback.",
          "Backtest M15 com Yahoo é limitado e deve ser validado com dados do broker quando possível."
        ]
      },
      "justificativa_tecnica": "Blue chip: marcada como viável por definição (robustez + liquidez estrutural)."
    },
    {
      "identificacao": {
        "nome": "VALE3",
        "codigo": "VALE3",
        "codigo_unico": "VALE3",
        "tipo": "AÇÃO",
        "categoria": "BLUE CHIP"
      },
      "ativo": {
        "category": "BLUE CHIP",
        "weight": 0.0,
        "ema_short": 12,
        "ema_long": 97,
        "rsi_low": 37,
        "rsi_high": 73,
        "adx_threshold": 13,
        "mom_min": 0.0,
        "sl_atr_multiplier": 1.9,
        "tp_mult": 3.0
      },
      "criterios_viabilidade": {
        "liquidez": {
          "metrica": "avg_fin_volume",
          "valor": 3156442.0932,
          "limiar": 318888.73997500003,
          "condicao": ">= limiar",
          "ok": true
        },
        "volatilidade": {
          "metrica": "volatility_ann",
          "valor": 0.17789100660644067,
          "limiar_inferior": 0.21543975635038212,
          "limiar_superior": 0.5751659600971757,
          "condicao": "entre [limiar_inferior, limiar_superior]",
          "ok": true
        },
        "correlacao_ibov": {
          "metrica": "abs_corr_ibov",
          "valor": 0.3644645128181709,
          "limiar": 0.6353184419932515,
          "condicao": "<= limiar",
          "ok": true
        },
        "market_cap": {
          "metrica": "market_cap",
          "valor": 336721346560.0,
          "limiar": 6132724479.999999,
          "condicao": ">= limiar (quando limiar > 0)",
          "ok": true
        },
        "regra_final": {
          "blue_chip": true,
          "criteria_passed": 4,
          "liquidez_obrigatoria": true,
          "min_criterios": 3
        }
      },
      "resultados": {
        "rank_total": 52,
        "tier": "D",
        "score_total": 0.28090909090909094,
        "liquidez_avg_fin_volume": 3156442.0932,
        "volatilidade_anualizada": 0.17789100660644067,
        "correlacao_ibov": 0.3644645128181709,
        "market_cap": 336721346560.0,
        "fonte_dados": "mt5"
      },
      "recomendacoes": {
        "acao_recomendada": "Adicionar ao universo de execução da próxima semana",
        "modo": "BLUE CHIP",
        "parametrizacao": "Usar ELITE_SYMBOLS (se existir) ou DEFAULT_PARAMS como base e refinar via otimização WFO.",
        "monitoramento": "Monitorar volatilidade e correlação diária; ajustar exposição se |corr| subir."
      },
      "prazos_e_recursos": {
        "prazo_estimado": "1 dia útil",
        "recursos_necessarios": [
          "MT5 conectado",
          "dados D1/M15",
          "cache Yahoo habilitado",
          "tempo de backtest"
        ]
      },
      "restricoes_dependencias": {
        "dependencias": [
          "MetaTrader5",
          "yfinance"
        ],
        "observacoes": [
          "Se o MT5 não tiver o símbolo, o script usa Yahoo como fallback.",
          "Backtest M15 com Yahoo é limitado e deve ser validado com dados do broker quando possível."
        ]
      },
      "justificativa_tecnica": "Blue chip: marcada como viável por definição (robustez + liquidez estrutural)."
    },
    {
      "identificacao": {
        "nome": "WEGE3",
        "codigo": "WEGE3",
        "codigo_unico": "WEGE3",
        "tipo": "AÇÃO",
        "categoria": "BLUE CHIP"
      },
      "ativo": {
        "category": "BLUE CHIP",
        "weight": 0.05,
        "ema_short": 9,
        "ema_long": 21,
        "rsi_low": 30.0,
        "rsi_high": 70.0,
        "adx_threshold": 25.0,
        "mom_min": 0.0,
        "sl_atr_multiplier": 2.5,
        "tp_mult": 5.0
      },
      "criterios_viabilidade": {
        "liquidez": {
          "metrica": "avg_fin_volume",
          "valor": 846845.6958500001,
          "limiar": 318888.73997500003,
          "condicao": ">= limiar",
          "ok": true
        },
        "volatilidade": {
          "metrica": "volatility_ann",
          "valor": 0.2752506917249384,
          "limiar_inferior": 0.21543975635038212,
          "limiar_superior": 0.5751659600971757,
          "condicao": "entre [limiar_inferior, limiar_superior]",
          "ok": true
        },
        "correlacao_ibov": {
          "metrica": "abs_corr_ibov",
          "valor": 0.1817572874467302,
          "limiar": 0.6353184419932515,
          "condicao": "<= limiar",
          "ok": true
        },
        "market_cap": {
          "metrica": "market_cap",
          "valor": 194344632320.0,
          "limiar": 6132724479.999999,
          "condicao": ">= limiar (quando limiar > 0)",
          "ok": true
        },
        "regra_final": {
          "blue_chip": true,
          "criteria_passed": 4,
          "liquidez_obrigatoria": true,
          "min_criterios": 3
        }
      },
      "resultados": {
        "rank_total": 54,
        "tier": "D",
        "score_total": 0.23181818181818184,
        "liquidez_avg_fin_volume": 846845.6958500001,
        "volatilidade_anualizada": 0.2752506917249384,
        "correlacao_ibov": 0.1817572874467302,
        "market_cap": 194344632320.0,
        "fonte_dados": "mt5"
      },
      "recomendacoes": {
        "acao_recomendada": "Adicionar ao universo de execução da próxima semana",
        "modo": "BLUE CHIP",
        "parametrizacao": "Usar ELITE_SYMBOLS (se existir) ou DEFAULT_PARAMS como base e refinar via otimização WFO.",
        "monitoramento": "Monitorar volatilidade e correlação diária; ajustar exposição se |corr| subir."
      },
      "prazos_e_recursos": {
        "prazo_estimado": "1 dia útil",
        "recursos_necessarios": [
          "MT5 conectado",
          "dados D1/M15",
          "cache Yahoo habilitado",
          "tempo de backtest"
        ]
      },
      "restricoes_dependencias": {
        "dependencias": [
          "MetaTrader5",
          "yfinance"
        ],
        "observacoes": [
          "Se o MT5 não tiver o símbolo, o script usa Yahoo como fallback.",
          "Backtest M15 com Yahoo é limitado e deve ser validado com dados do broker quando possível."
        ]
      },
      "justificativa_tecnica": "Blue chip: marcada como viável por definição (robustez + liquidez estrutural)."
    },
    {
      "identificacao": {
        "nome": "AURA33",
        "codigo": "AURA33",
        "codigo_unico": "AURA33",
        "tipo": "BDR",
        "categoria": "OPORTUNIDADE"
      },
      "ativo": {
        "category": "OPORTUNIDADE",
        "weight": 0.05,
        "ema_short": 9,
        "ema_long": 21,
        "rsi_low": 30.0,
        "rsi_high": 70.0,
        "adx_threshold": 25.0,
        "mom_min": 0.0,
        "sl_atr_multiplier": 2.5,
        "tp_mult": 5.0
      },
      "criterios_viabilidade": {
        "liquidez": {
          "metrica": "avg_fin_volume",
          "valor": 1806209.4915000005,
          "limiar": 318888.73997500003,
          "condicao": ">= limiar",
          "ok": true
        },
        "volatilidade": {
          "metrica": "volatility_ann",
          "valor": 0.4880879902923411,
          "limiar_inferior": 0.21543975635038212,
          "limiar_superior": 0.5751659600971757,
          "condicao": "entre [limiar_inferior, limiar_superior]",
          "ok": true
        },
        "correlacao_ibov": {
          "metrica": "abs_corr_ibov",
          "valor": 0.05689552751983074,
          "limiar": 0.6353184419932515,
          "condicao": "<= limiar",
          "ok": true
        },
        "market_cap": {
          "metrica": "market_cap",
          "valor": 27092271104.0,
          "limiar": 6132724479.999999,
          "condicao": ">= limiar (quando limiar > 0)",
          "ok": true
        },
        "regra_final": {
          "blue_chip": false,
          "criteria_passed": 4,
          "liquidez_obrigatoria": true,
          "min_criterios": 3
        }
      },
      "resultados": {
        "rank_total": 55,
        "tier": "D",
        "score_total": 0.18727272727272726,
        "liquidez_avg_fin_volume": 1806209.4915000005,
        "volatilidade_anualizada": 0.4880879902923411,
        "correlacao_ibov": 0.05689552751983074,
        "market_cap": 27092271104.0,
        "fonte_dados": "mt5"
      },
      "recomendacoes": {
        "acao_recomendada": "Adicionar ao universo de execução da próxima semana",
        "modo": "OPORTUNIDADE",
        "parametrizacao": "Usar ELITE_SYMBOLS (se existir) ou DEFAULT_PARAMS como base e refinar via otimização WFO.",
        "monitoramento": "Monitorar volatilidade e correlação diária; ajustar exposição se |corr| subir."
      },
      "prazos_e_recursos": {
        "prazo_estimado": "1 dia útil",
        "recursos_necessarios": [
          "MT5 conectado",
          "dados D1/M15",
          "cache Yahoo habilitado",
          "tempo de backtest"
        ]
      },
      "restricoes_dependencias": {
        "dependencias": [
          "MetaTrader5",
          "yfinance"
        ],
        "observacoes": [
          "Se o MT5 não tiver o símbolo, o script usa Yahoo como fallback.",
          "Backtest M15 com Yahoo é limitado e deve ser validado com dados do broker quando possível."
        ]
      },
      "justificativa_tecnica": "Liquidez OK (R$ 1.806.209; lim=R$ 318.889); Vol OK (48,81%; faixa=21,54%–57,52%); Corr OK (|corr|=0,057; lim=0,635); MCap OK (R$ 27.092.271.104; lim=R$ 6.132.724.480)"
    }
  ]
}"""
ATIVOS_VIAVEIS_REPORT = json.loads(ATIVOS_VIAVEIS_REPORT_JSON)

ATIVOS_VIAVEIS_REPORT = json.loads(ATIVOS_VIAVEIS_REPORT_JSON)

ATIVOS_VIAVEIS_REPORT = json.loads(ATIVOS_VIAVEIS_REPORT_JSON)

ATIVOS_VIAVEIS_REPORT = json.loads(ATIVOS_VIAVEIS_REPORT_JSON)

ATIVOS_VIAVEIS_REPORT = json.loads(ATIVOS_VIAVEIS_REPORT_JSON)

ATIVOS_VIAVEIS_REPORT = json.loads(ATIVOS_VIAVEIS_REPORT_JSON)

ATIVOS_VIAVEIS_REPORT = json.loads(ATIVOS_VIAVEIS_REPORT_JSON)

ATIVOS_VIAVEIS_REPORT = json.loads(ATIVOS_VIAVEIS_REPORT_JSON)

ATIVOS_VIAVEIS_REPORT = json.loads(ATIVOS_VIAVEIS_REPORT_JSON)

ATIVOS_VIAVEIS_REPORT = json.loads(ATIVOS_VIAVEIS_REPORT_JSON)

ATIVOS_VIAVEIS_REPORT = json.loads(ATIVOS_VIAVEIS_REPORT_JSON)

ATIVOS_VIAVEIS_REPORT = json.loads(ATIVOS_VIAVEIS_REPORT_JSON)
