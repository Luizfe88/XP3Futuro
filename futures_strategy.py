# futures_strategy.py
import logging
from datetime import datetime

logger = logging.getLogger("futures_strategy")

def get_session_params(current_time=None):
    """
    Retorna parâmetros de ADX baseados na sessão (Prioridade 3).
    Abertura: 28, Meio-dia: 22, Tarde: 25.
    """
    if current_time is None:
        current_time = datetime.now().time()
    
    # ABERTURA (09:00 - 11:30)
    if current_time < datetime.strptime("11:30", "%H:%M").time():
        return {"adx_min": 28, "session": "ABERTURA"}
    
    # MEIO_DIA (11:30 - 14:30)
    elif current_time < datetime.strptime("14:30", "%H:%M").time():
        return {"adx_min": 22, "session": "MEIO_DIA"}
    
    # TARDE (14:30 - 17:30)
    else:
        return {"adx_min": 25, "session": "TARDE"}

def validate_futures_entry(adx_value, current_time=None):
    """Valida entrada baseada no ADX da sessão"""
    params = get_session_params(current_time)
    if adx_value < params["adx_min"]:
        return False, f"ADX {adx_value:.1f} < {params['adx_min']} ({params['session']})"
    return True, "OK"
