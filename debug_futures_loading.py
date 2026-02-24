
import logging
import pandas as pd
import otimizador_semanal
import config_futures
import MetaTrader5 as mt5

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("debug_futures")

def test_loading():
    symbol = "WING26" # Using a liquid one
    print(f"Testing data loading for: {symbol}")
    
    # Check MT5 connection explicitely
    if not mt5.initialize():
        print("MT5 Initialize failed in test script")
        print("Error:", mt5.last_error())
    else:
        print("MT5 Initialized successfully")
        print("Terminal Info:", mt5.terminal_info())
        
        # Check if symbol exists
        info = mt5.symbol_info(symbol)
        if info:
            print(f"Symbol {symbol} found. Selectable: {info.select}")
            if not info.select:
                print("Selecting symbol...")
                mt5.symbol_select(symbol, True)
        else:
            print(f"Symbol {symbol} NOT found in MT5!")
            
            # List some symbols to see what's available
            print("Listing first 10 symbols in MT5:")
            symbols = mt5.symbols_get()
            if symbols:
                for s in symbols[:10]:
                    print(s.name)
            else:
                print("No symbols found in MT5?")

    # Now test the function
    print("\nCalling load_data_with_retry...")
    df = otimizador_semanal.load_data_with_retry(symbol, 1000)
    
    if df is not None and not df.empty:
        print(f"SUCCESS! Loaded {len(df)} rows.")
        print(df.tail())
    else:
        print("FAILURE: Data is None or empty.")

if __name__ == "__main__":
    test_loading()
