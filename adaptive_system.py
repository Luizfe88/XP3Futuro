"""
Módulo responsável pela lógica do Sistema Adaptativo de 4 Camadas.
"""
import time
from collections import deque
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
import logging
import os

# =============================================================================
# SETUP DE LOGGING DETALHADO
# =============================================================================
def setup_adaptive_logger():
    """Cria um logger dedicado para o sistema adaptativo."""
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    logger = logging.getLogger("adaptive_system")
    logger.setLevel(logging.INFO)
    
    # Evita adicionar handlers duplicados
    if logger.hasHandlers():
        logger.handlers.clear()
        
    handler = logging.FileHandler(os.path.join(log_dir, "adaptive_system.log"), mode='a', encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    return logger

adaptive_logger = setup_adaptive_logger()
# =============================================================================

import utils
import config

# =============================================================================
# 1. CAMADA SENSOR: Coleta de Métricas
# =============================================================================

_sensor_data_cache = {
    "last_collection_time": 0,
    "metrics": {
        "volatility": {"atr_d1": 0.0, "atr_m15": 0.0, "ratio": 0.0},
        "relative_volume": {"rvol": 0.0, "avg_rvol": 0.0},
        "recent_performance": {"pnl": 0.0, "win_rate": 0.0, "drawdown": 0.0},
    }
}

def collect_sensor_data(force_run=False):
    """
    Coleta e armazena na memória os dados do mercado a cada 15 minutos.
    - Volatilidade (ATR)
    - Volume Relativo (RVOL)
    - Performance Recente (P&L de Curto Prazo)
    """
    now = time.time()
    if not force_run and (now - _sensor_data_cache["last_collection_time"] < 900): # 15 minutos
        return _sensor_data_cache["metrics"]

    adaptive_logger.info("🤖 SENSOR: Coletando métricas de mercado...")

    try:
        # 1. Volatilidade (ATR)
        atr_d1 = _calculate_average_atr_d1()
        atr_m15 = _calculate_current_atr_m15()
        volatility_ratio = (atr_m15 / atr_d1) if atr_d1 > 0 else 1.0

        _sensor_data_cache["metrics"]["volatility"] = {
            "atr_d1": atr_d1,
            "atr_m15": atr_m15,
            "ratio": volatility_ratio
        }

        # 2. Volume Relativo (RVOL)
        rvol, avg_rvol = _calculate_rvol("IBOV")
        _sensor_data_cache["metrics"]["relative_volume"] = {
            "rvol": rvol,
            "avg_rvol": avg_rvol
        }

        # 3. Performance Recente (P&L de Curto Prazo)
        pnl_2h, win_rate_2h, max_dd_2h = _calculate_recent_performance()
        _sensor_data_cache["metrics"]["recent_performance"] = {
            "pnl": pnl_2h,
            "win_rate": win_rate_2h,
            "drawdown": max_dd_2h
        }

        _sensor_data_cache["last_collection_time"] = now
        adaptive_logger.info(f"🤖 SENSOR: Métricas atualizadas. Volatility Ratio: {volatility_ratio:.2f}")

    except Exception as e:
        adaptive_logger.error(f"Erro na coleta de dados do SENSOR: {e}", exc_info=True)

    return _sensor_data_cache["metrics"]


def _calculate_average_atr_d1(symbol="IBOV", period=14, days=5):
    """Calcula o ATR médio dos últimos 'days' dias para o timeframe D1."""
    df_d1 = utils.safe_copy_rates(symbol, mt5.TIMEFRAME_D1, days + period)
    if df_d1 is None or len(df_d1) < days + period:
        return 0.01 # Retorna um valor padrão pequeno para evitar divisão por zero

    atrs = []
    for i in range(days):
        # Janela de 'period' dias para cada cálculo de ATR
        window = df_d1.iloc[i : i + period]
        atr = utils.get_atr(window, period=period)
        if atr:
            atrs.append(atr)

    return np.mean(atrs) if atrs else 0.01


def _calculate_current_atr_m15(symbol="IBOV", period=14):
    """Calcula o ATR atual no timeframe M15."""
    df_m15 = utils.safe_copy_rates(symbol, mt5.TIMEFRAME_M15, period * 2)
    if df_m15 is None or len(df_m15) < period:
        return 0.0

    return utils.get_atr(df_m15, period=period)


# =============================================================================
# 2. CAMADA CÉREBRO: Análise de Regime
# =============================================================================

current_regime = "NEUTRAL"  # Variável global para acesso externo

def analyze_market_regime():
    """
    Analisa os dados da camada Sensor para detectar o regime de mercado.
    Retorna: "TREND", "REVERSION" ou "NEUTRAL"
    """
    global current_regime
    
    metrics = _sensor_data_cache["metrics"]
    volatility_ratio = metrics["volatility"].get("ratio", 1.0)

    # Lógica de decisão (exemplo inicial)
    if volatility_ratio > 1.2:  # Ajustado de 1.5 para 1.2 (mais sensível)
        current_regime = "TREND"
    elif volatility_ratio < 0.8:
        current_regime = "REVERSION"
    else:
        current_regime = "NEUTRAL"
        
    return current_regime

# =============================================================================
# 3. CAMADA MECÂNICO: Ajuste de Parâmetros
# =============================================================================

def adjust_parameters(regime):
    """
    Ajusta os parâmetros de trading com base no regime de mercado detectado.
    Modifica os parâmetros otimizados globalmente em tempo real.
    """
    adaptive_logger.info(f"🔧 MECÂNICO: Ajustando parâmetros para o regime '{regime}'...")
    
    # Importar o dicionário de parâmetros otimizados do bot.py
    import bot
    
    if not hasattr(bot, 'optimized_params') or not bot.optimized_params:
        adaptive_logger.warning("🔧 MECÂNICO: Parâmetros otimizados não encontrados, usando configurações padrão")
        return
    
    # Aplicar ajustes baseado no regime
    for symbol in bot.optimized_params:
        params = bot.optimized_params[symbol]
        
        if regime == "TREND":
            # Mercado em tendência: usar configurações mais agressivas
            if isinstance(params, dict):
                if "parameters" in params:
                    params["parameters"]["adx_threshold"] = config.ADAPTIVE_THRESHOLDS["RISK_ON"]["min_adx"]
                else:
                    params["adx_threshold"] = config.ADAPTIVE_THRESHOLDS["RISK_ON"]["min_adx"]
                    
        elif regime == "REVERSION":
            # Mercado lateral/reversão: usar configurações mais conservadoras
            if isinstance(params, dict):
                if "parameters" in params:
                    params["parameters"]["adx_threshold"] = config.ADAPTIVE_THRESHOLDS["RISK_OFF"]["min_adx"]
                else:
                    params["adx_threshold"] = config.ADAPTIVE_THRESHOLDS["RISK_OFF"]["min_adx"]
                    
        else: # NEUTRAL
            # Voltar aos padrões (não altera os parâmetros otimizados)
            pass
    
    adaptive_logger.info(f"🔧 MECÂNICO: Parâmetros ajustados para regime '{regime}' em {len(bot.optimized_params)} símbolos")

    # Log dos parâmetros modificados para debug
    sample_params = {}
    if bot.optimized_params:
        # Pega os parâmetros do primeiro símbolo como exemplo
        first_symbol = list(bot.optimized_params.keys())[0]
        sample_params = bot.optimized_params[first_symbol]
        
    adaptive_logger.info(f"🔧 MECÂNICO: Exemplo de parâmetros ajustados (simulação): {sample_params}")


# =============================================================================
# 4. CAMADA EVOLUÇÃO: Feedback Loop
# =============================================================================

_vaccine_cache = {} # Formato: {"SYMBOL": expiration_timestamp}

def apply_vaccine(symbol, reason, duration_hours=2):
    """
    Aplica uma "vacina" temporária a um ativo após um stop loss específico,
    penalizando-o para futuras entradas.
    """
    if "slippage" in reason.lower() or "spread" in reason.lower():
        expiration = time.time() + duration_hours * 3600
        _vaccine_cache[symbol] = expiration
        adaptive_logger.warning(f"💉 EVOLUÇÃO: Vacina de slippage aplicada a {symbol}. Expira em {duration_hours}h.")

def is_vaccinated(symbol):
    """
    Verifica se um ativo está atualmente "vacinado".
    Remove vacinas expiradas.
    """
    if symbol not in _vaccine_cache:
        return False
    
    now = time.time()
    if now > _vaccine_cache[symbol]:
        adaptive_logger.info(f"💉 EVOLUÇÃO: Vacina para {symbol} expirou. Removendo penalidade.")
        del _vaccine_cache[symbol]
        return False
        
    return True


# =============================================================================
# 🚨 GATILHO DE PÂNICO (CIRCUIT BREAKER)
# =============================================================================

def check_panic_mode():
    """
    Verifica condições de pânico e força ajustes imediatos se necessário.
    """
    # 1. Queda brusca do índice
    ibov = utils.safe_copy_rates("IBOV", mt5.TIMEFRAME_M1, 10)
    if ibov is not None and len(ibov) == 10:
        price_start = ibov['open'].iloc[0]
        price_end = ibov['close'].iloc[-1]
        change_pct = ((price_end - price_start) / price_start) * 100
        if change_pct < -1.0:
            adaptive_logger.critical(f"🚨 PANIC MODE: Queda de {change_pct:.2f}% no IBOV em 10 min!")
            # Forçar imediatamente o modo RISK_OFF
            adjust_parameters("REVERSION") 
            return True

    # 2. Surto de Liquidez (RVOL)
    # A lógica de RVOL precisa ser implementada primeiro.

    return False

# =============================================================================
# Funções Auxiliares para Cálculos do Sensor
# =============================================================================

def _calculate_rvol(symbol="IBOV", period=20):
    """
    Calcula o Volume Relativo (RVOL) atual vs média.
    """
    try:
        df_m15 = utils.safe_copy_rates(symbol, mt5.TIMEFRAME_M15, period * 2)
        if df_m15 is None or len(df_m15) < period * 2:
            return 1.0, 1.0

        # Calcula volume médio das últimas 'period' barras
        current_volume = df_m15['tick_volume'].tail(period).mean()
        avg_volume = df_m15['tick_volume'].rolling(window=period).mean().iloc[-period-1:].mean()

        rvol = (current_volume / avg_volume) if avg_volume > 0 else 1.0
        return rvol, avg_volume
    except Exception as e:
        adaptive_logger.error(f"Erro ao calcular RVOL para {symbol}: {e}")
        return 1.0, 1.0

def _calculate_recent_performance(lookback_hours=2):
    """
    Calcula a performance recente (P&L, Win Rate, Drawdown) das últimas 'lookback_hours' horas.
    Por enquanto, retorna valores padrão. A implementação completa requer acesso ao histórico de trades.
    """
    try:
        # Placeholder: Implementar lógica para calcular P&L, Win Rate e Drawdown
        # Isso requer acesso ao histórico de trades fechados nas últimas X horas.
        # Por enquanto, retornamos valores neutros.
        pnl_2h = 0.0
        win_rate_2h = 0.5 # 50% de win rate padrão
        max_dd_2h = 0.0
        return pnl_2h, win_rate_2h, max_dd_2h
    except Exception as e:
        adaptive_logger.error(f"Erro ao calcular performance recente: {e}")
        return 0.0, 0.5, 0.0

