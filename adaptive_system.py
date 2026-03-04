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
    "metrics": {}, # Dicionário de métricas por ativo: {"WIN": {...}, "WDO": {...}}
    "global_regime": "NEUTRAL"
}

def collect_sensor_data(symbols_to_scan=None, force_run=False):
    """
    Coleta e armazena na memória os dados do mercado a cada 15 minutos.
    Agora suporta análise individual por ativo.
    """
    now = time.time()
    if not force_run and (now - _sensor_data_cache["last_collection_time"] < 900): # 15 minutos
        return _sensor_data_cache["metrics"]

    adaptive_logger.info("🤖 SENSOR: Coletando métricas de mercado por ativo...")
    
    # Lista padrão se não fornecida
    if not symbols_to_scan:
        # Usa utils.resolve_symbol() para garantir símbolos negociáveis
        # em vez de genéricos que podem falhar no MT5
        base_assets = ["WIN", "WDO", "IND", "DOL", "WSP", "BGI", "ICF", "CCM", "BIT"]
        symbols_to_scan = []
        for base in base_assets:
            real = utils.resolve_symbol(base)
            if real:
                symbols_to_scan.append(real)
            else:
                # Fallback se não achar contrato vigente (usa $N)
                symbols_to_scan.append(f"{base}$N")
    
    # Remove duplicatas e garante lista limpa
    symbols_to_scan = list(set(symbols_to_scan))

    metrics_map = {}

    try:
        for base in symbols_to_scan:
            # ✅ CORREÇÃO CRÍTICA: Converter para contrato real ANTES do MT5
            # Isso evita erros de OHLC inválido com símbolos genéricos
            real_sym = utils.resolve_symbol(base)
            
            if not real_sym:
                # Tenta usar o próprio base se a resolução falhar
                real_sym = base
                
            adaptive_logger.info(f"🔄 Coletando métricas para {base} → {real_sym}")
            
            # 1. Volatilidade (ATR)
            atr_d1 = _calculate_average_atr_d1(real_sym)
            atr_m15 = _calculate_current_atr_m15(real_sym)
            
            # Normalização do Ratio: ATR_D1 / 6.0 (aprox. desvio padrão diário para intraday)
            volatility_ratio = (atr_m15 / (atr_d1 / 6.0)) if atr_d1 > 0 else 1.0

            # 2. Volume Relativo (RVOL)
            rvol, avg_rvol = _calculate_rvol(real_sym)
            
            # Armazena usando a chave original (para compatibilidade) OU o símbolo real?
            # Melhor usar o símbolo real para garantir consistência no adjust_parameters
            metrics_data = {
                "volatility": {
                    "atr_d1": atr_d1,
                    "atr_m15": atr_m15,
                    "ratio": volatility_ratio
                },
                "relative_volume": {
                    "rvol": rvol,
                    "avg_rvol": avg_rvol
                }
            }
            metrics_map[real_sym] = metrics_data
            
            # ✅ COMPATIBILIDADE: Salva também como WIN$N/WDO$N para analyze_market_regime()
            # pois a função de regime busca o padrão genérico se não especificado
            if "WIN" in base or "WIN" in real_sym:
                 metrics_map["WIN$N"] = metrics_data
            elif "WDO" in base or "WDO" in real_sym:
                 metrics_map["WDO$N"] = metrics_data
            
            # adaptive_logger.debug(f"   📊 {real_sym}: VolRatio={volatility_ratio:.2f} RVOL={rvol:.2f}")

        # Atualiza cache
        _sensor_data_cache["metrics"] = metrics_map
        _sensor_data_cache["last_collection_time"] = now
        adaptive_logger.info(f"🤖 SENSOR: Métricas atualizadas para {len(metrics_map)} ativos.")

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
# 2. CAMADA CÉREBRO: Análise de Regime (INDIVIDUALIZADA)
# =============================================================================

def analyze_market_regime(symbol="WIN$N"):
    """
    Analisa os dados da camada Sensor para detectar o regime de mercado de um ATIVO ESPECÍFICO.
    Retorna: "TREND", "REVERSION" ou "NEUTRAL"
    """
    metrics_map = _sensor_data_cache.get("metrics", {})
    
    # Se não tiver dados específicos, tenta usar dados globais ou retorna NEUTRAL
    if not metrics_map or symbol not in metrics_map:
        # Fallback: Tenta achar WIN ou WDO se o símbolo for genérico
        fallback = metrics_map.get("WIN$N") or metrics_map.get("WDO$N")
        if not fallback:
            return "NEUTRAL"
        metrics = fallback
    else:
        metrics = metrics_map[symbol]

    volatility_ratio = metrics["volatility"].get("ratio", 1.0)

    # Lógica de decisão
    if volatility_ratio > 1.2:
        return "TREND"
    elif volatility_ratio < 0.8:
        return "REVERSION"
    else:
        return "NEUTRAL"

# =============================================================================
# 3. CAMADA MECÂNICO: Ajuste de Parâmetros
# =============================================================================

def adjust_parameters(force_regime=None):
    """
    Ajusta os parâmetros de trading para CADA ATIVO com base no seu regime.
    Modifica os parâmetros otimizados globalmente em tempo real.
    """
    # Importar o dicionário de parâmetros otimizados do bot.py
    import bot
    
    if not hasattr(bot, 'optimized_params') or not bot.optimized_params:
        return
    
    adjusted_count = 0
    
    for symbol in bot.optimized_params:
        # Detecta regime individual (ou usa o forçado se teste/pânico)
        regime = force_regime if force_regime else analyze_market_regime(symbol)
        
        params = bot.optimized_params[symbol]
        
        # Define perfil de risco baseado no regime
        if regime == "TREND":
            # Mercado em tendência: Risk ON
            target_adx = config.ADAPTIVE_THRESHOLDS["RISK_ON"]["min_adx"]
        elif regime == "REVERSION":
            # Mercado lateral: Risk OFF
            target_adx = config.ADAPTIVE_THRESHOLDS["RISK_OFF"]["min_adx"]
        else:
            # NEUTRAL: Mantém original otimizado (não altera)
            continue
            
        # Aplica o ajuste
        if isinstance(params, dict):
            if "parameters" in params:
                params["parameters"]["adx_threshold"] = target_adx
            else:
                params["adx_threshold"] = target_adx
        
        adjusted_count += 1
    
    if adjusted_count > 0:
        adaptive_logger.info(f"🔧 MECÂNICO: Parâmetros ajustados individualmente para {adjusted_count} ativos.")


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
    # Pega o RVOL do Índice Futuro, que reflete o apetite/pânico instantâneo mais que o mercado à vista
    rvol, _ = _calculate_rvol("WIN$N", period=10)
    if rvol > 3.5:
        adaptive_logger.critical(f"🚨 PANIC MODE: Surto de liquidez detectado! RVOL = {rvol:.2f}x a média.")
        # Força proteção imediata contra volatilidade predatória
        adjust_parameters("REVERSION")
        return True

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

