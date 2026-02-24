
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict

def get_historical_data(symbol: str, days: int = 20) -> Optional[pd.DataFrame]:
    """
    Busca um histórico de dados diários para um ativo.
    """
    try:
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_D1, 0, days)
        if rates is None or len(rates) == 0:
            return None
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df
    except Exception as e:
        print(f"Erro ao buscar dados históricos para {symbol}: {e}")
        return None


def get_live_spread(symbol: str) -> Optional[float]:
    """
    Calcula o spread atual (bid/ask) de um ativo.
    """
    try:
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return None
        
        bid = tick.bid
        ask = tick.ask

        if bid > 0 and ask > 0:
            return ask - bid
        
        return None
    except Exception as e:
        print(f"Erro ao buscar spread para {symbol}: {e}")
        return None


