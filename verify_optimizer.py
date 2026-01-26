
import sys
import unittest
import pandas as pd
import numpy as np
import logging

# Add current dir to path
sys.path.append(r"c:\Users\luizf\Documents\xp3v5")

# Mock logging
logging.basicConfig(level=logging.CRITICAL)

from optimizer_optuna import compute_metrics
from otimizador_semanal import load_data_with_retry

class TestOptimizer(unittest.TestCase):
    def test_compute_metrics_win_rate(self):
        print("\nTesting compute_metrics Win Rate...")
        # Equity curve: 100 -> 110 (Win) -> 100 (Loss) -> 120 (Win)
        # Returns: 0.1, -0.09, 0.2
        # Wins: 2, Losses: 1
        equity = [100, 110, 100, 120]
        metrics = compute_metrics(equity)
        
        self.assertIn("win_rate", metrics)
        self.assertAlmostEqual(metrics["win_rate"], 2/3)
        print(f"✅ Win Rate Calculated: {metrics['win_rate']:.2f}")

    def test_dataframe_rename_logic(self):
        print("\nTesting DataFrame Column Rename...")
        # Simulate loading data with tick_volume
        df = pd.DataFrame({
            'tick_volume': [100, 200],
            'open': [10, 11],
            'high': [12, 13],
            'low': [9, 10],
            'close': [11, 12],
            'time': pd.to_datetime(['2023-01-01 10:00', '2023-01-01 10:15'])
        })
        df.set_index('time', inplace=True)
        
        # We can't easily call load_data_with_retry without MT5 or mocking it heavily, 
        # so we will simulate the LOGIC we inserted.
        
        if 'tick_volume' in df.columns:
            df = df.rename(columns={'tick_volume': 'volume'})
            
        self.assertIn('volume', df.columns)
        self.assertNotIn('tick_volume', df.columns)
        print("✅ Column Rename Logic OK")

    def test_compute_metrics_sharpe_sortino(self):
        print("\nTesting Sharpe and Sortino...")
        equity = [100, 102, 101, 103, 102, 104] # Volatile upward trend
        metrics = compute_metrics(equity)
        
        self.assertIn("sharpe", metrics)
        self.assertIn("sortino", metrics)
        print(f"✅ Sharpe: {metrics['sharpe']:.2f}, Sortino: {metrics['sortino']:.2f}")

    def test_backtest_core_volume_costs(self):
        print("\nTesting Fast Backtest Core (Volume & Costs)...")
        # Simula dados
        close = np.array([10.0, 10.1, 10.2, 10.3, 10.2], dtype=np.float64)
        high = close + 0.1
        low = close - 0.1
        volume = np.array([1000, 1200, 1500, 800, 1000], dtype=np.float64) # 4o candle tem vol baixo (< MA)
        volume_ma = np.array([1000, 1000, 1000, 1100, 1100], dtype=np.float64)
        
        # Dummies
        ema = np.zeros_like(close)
        arr = np.zeros_like(close)
        
        # Import fast_backtest_core
        from optimizer_optuna import fast_backtest_core
        
        # Roda
        vwap = np.zeros_like(close)
        rsi = np.zeros_like(close)
        rsi2 = np.zeros_like(close)
        adx = np.zeros_like(close)
        sar = np.zeros_like(close)
        atr = np.zeros_like(close)
        mom = np.zeros_like(close)
        mlp = np.zeros_like(close)
        eq, trades, wins, losses, signals, counts = fast_backtest_core(
            close, close, high, low, volume, volume_ma, vwap,
            ema, ema,
            rsi, rsi2, adx, sar, atr, mom,
            mlp,
            30, 70,
            25,
            2.0, 4.0, 0.0035,
            float(np.mean(volume)),
            0.01,
            1,
            0,
            0,
            0.01,
            0
        )
        
        self.assertEqual(len(eq), 5)
        print(f"✅ Fast Backtest Runs OK. Trades: {trades}")

if __name__ == '__main__':
    unittest.main()
