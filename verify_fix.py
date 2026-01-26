import pandas as pd
import numpy as np
import logging
import sys

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.append(r"c:\Users\luizf\Documents\xp3v5")

def verify_fixes():
    try:
        print("1. Importing optimizer_optuna...")
        from optimizer_optuna import backtest_params_on_df, fast_backtest_core
        print(" [OK] Import Successful")
        
        print("2. Creating Dummy Data...")
        dates = pd.date_range(start="2023-01-01", periods=200, freq="15min")
        df = pd.DataFrame({
            "open": np.random.uniform(10, 20, 200).astype(float),
            "high": np.random.uniform(20, 30, 200).astype(float),
            "low": np.random.uniform(5, 10, 200).astype(float),
            "close": np.random.uniform(10, 20, 200).astype(float),
            "volume": np.random.uniform(1000, 5000, 200).astype(float),
        }, index=dates)
        
        params = {
            "ema_short": 9, "ema_long": 21,
            "rsi_low": 30, "rsi_high": 70,
            "adx_threshold": 25,
            "sl_atr_multiplier": 2.0,
            "tp_mult": 4.0,
            "base_slippage": 0.001
        }
        
        print("3. Running Backtest (Numba Compilation Check)...")
        # This triggers fast_backtest_core compilation
        metrics = backtest_params_on_df("DUMMY", params, df, ml_model=None)
        
        print(f" [OK] Backtest Metrics: {metrics}")
        
        print("4. Checking otimizador_semanal imports...")
        import otimizador_semanal
        print(" [OK] Otimizador Semanal Imported")
        
        print("ALL CHECKS PASSED")
        return True
        
    except Exception as e:
        print(f"FAILURE: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    try:
        if sys.platform.startswith('win'):
            import os
            os.system('chcp 65001')
    except:
        pass
    verify_fixes()
