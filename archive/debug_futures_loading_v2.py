
import logging
import pandas as pd
import otimizador_semanal
import MetaTrader5 as mt5
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("debug_futures_v2")

def test_loading():
    symbol = "WING26" 
    print(f"Testing SPECIALIZED futures data loading for: {symbol}")
    
    # Initialize MT5 manually to be sure
    if not mt5.initialize():
        print("MT5 Init failed")
        return

    # 1. Test load_futures_data_for_optimizer
    print("\n[TEST 1] Calling load_futures_data_for_optimizer...")
    try:
        df = otimizador_semanal.load_futures_data_for_optimizer(symbol, 1000, "M15")
        if df is not None and not df.empty:
            print(f"SUCCESS! Loaded {len(df)} rows.")
            print(df.tail())
        else:
            print("FAILURE: load_futures_data_for_optimizer returned None/Empty.")
    except Exception as e:
        print(f"EXCEPTION in load_futures_data_for_optimizer: {e}")

    # 2. Test direct copy again with error printing
    print("\n[TEST 2] Direct copy_rates_from_pos for WING26...")
    rates = mt5.copy_rates_from_pos("WING26", mt5.TIMEFRAME_M15, 0, 1000)
    if rates is None:
        print(f"Direct copy failed. Last Error: {mt5.last_error()}")
    else:
        print(f"Direct copy SUCCESS. Rows: {len(rates)}")
        
    # 3. Test generic load_data_with_retry (current implementation)
    print("\n[TEST 3] Calling load_data_with_retry...")
    df2 = otimizador_semanal.load_data_with_retry(symbol, 1000, "M15")
    if df2 is not None and not df2.empty:
        print(f"SUCCESS (Generic)! Loaded {len(df2)} rows.")
    else:
        print("FAILURE (Generic).")

if __name__ == "__main__":
    test_loading()
