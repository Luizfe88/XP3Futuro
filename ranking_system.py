import math
import logging

logger = logging.getLogger(__name__)

def calculate_trigger_distance_score(ind_data: dict, current_price: float, side: str) -> float:
    """
    10% peso - Distância do gatilho (VWAP ou EMA).
    Retorna 0.0 a 1.0 (onde 1.0 = muito perto do breakout/VWAP/EMA).
    """
    vwap = float(ind_data.get("vwap", current_price))
    ema_fast = float(ind_data.get("ema_fast", current_price))
    
    if current_price <= 0:
        return 0.5
    
    dist_vwap = abs(current_price - vwap) / current_price
    dist_ema = abs(current_price - ema_fast) / current_price
    
    # Menor distância = maior pontuação
    min_dist = min(dist_vwap, dist_ema)
    
    # Assumindo que 0.5% (0.005) de distância é o 'longe' e 0% é o 'colado' no gatilho.
    # Clip entre 0 e 1.
    score = max(0.0, 1.0 - (min_dist / 0.005))
    return score

def calculate_ema_rsi_alignment(ind_data: dict, side: str) -> float:
    """
    15% peso - Alinhamento EMA + RSI saudável.
    """
    ema_fast = float(ind_data.get("ema_fast", 0))
    ema_slow = float(ind_data.get("ema_slow", 0))
    rsi = float(ind_data.get("rsi", 50))
    
    score = 0.0
    
    # Alinhamento básico de médias
    if (side == "BUY" and ema_fast > ema_slow) or (side == "SELL" and ema_fast < ema_slow):
        score += 0.5
        
    # RSI Saudável (entre 40 e 68 para compra, 32 e 60 para venda)
    if side == "BUY" and 40 <= rsi <= 68:
        score += 0.5
    elif side == "SELL" and 32 <= rsi <= 60:
        score += 0.5
        
    return score

def rank_opportunities(scanned_indicators: dict) -> list:
    """
    Calcula um ranking completo de todos os símbolos usando:
    - Score real (peso 40%)
    - ADX (peso 20%)
    - Volume relativo (peso 15%)
    - Alinhamento EMA + RSI saudável (15%)
    - Distância do gatilho (peso 10%)
    
    Retorna a lista ordenada do melhor para o pior.
    """
    ranked_list = []
    
    if not scanned_indicators:
        return []

    for symbol, ind_data in scanned_indicators.items():
        if not ind_data or ind_data.get("error"):
            continue
            
        base_score = float(ind_data.get("score", 0))
        adx = float(ind_data.get("adx", 0))
        vol_ratio = float(ind_data.get("volume_ratio", 0))
        
        # RELAXADO: Aceita qualquer ativo para o ranking, mesmo com score 0
        # A pontuação final vai diferenciar os melhores.
        if base_score <= 0:
            base_score = 0.1
            
        ema_fast = float(ind_data.get("ema_fast", 0))
        ema_slow = float(ind_data.get("ema_slow", 0))
        side = "BUY" if ema_fast > ema_slow else "SELL"
        
        # 1. Score real (Max 100) -> 40%
        norm_score = min(base_score / 100.0, 1.0)
        
        # 2. ADX (Max 50) -> 20%
        norm_adx = min(adx / 50.0, 1.0)
        
        # 3. Volume Reativo (Max 3.0x média) -> 15%
        norm_vol = min(vol_ratio / 3.0, 1.0)
        
        # 4. Alinhamento -> 15%
        align_score = calculate_ema_rsi_alignment(ind_data, side)
        
        # 5. Distância Gatilho -> 10%
        close_price = float(ind_data.get("close", 0))
        trigger_score = calculate_trigger_distance_score(ind_data, close_price, side)
        
        # Peso Final (escala 0 a 100)
        final_score = (
            (norm_score * 0.40) +
            (norm_adx * 0.20) +
            (norm_vol * 0.15) +
            (align_score * 0.15) +
            (trigger_score * 0.10)
        ) * 100.0
        
        ranked_list.append({
            "symbol": symbol,
            "side": side,
            "final_score": final_score,
            "base_score": base_score,
            "adx": adx,
            "vol_ratio": vol_ratio,
            "ind_data": ind_data
        })
        
    # Ordenar do melhor (maior score final) para o pior
    ranked_list.sort(key=lambda x: x["final_score"], reverse=True)
    return ranked_list
