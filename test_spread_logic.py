import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
from datetime import datetime, timedelta

import utils
import daily_analysis_logger
utils.TIMEFRAME_MACRO = 60
utils.mt5.TIMEFRAME_H1 = 60

def make_df(n=300, base_close=30.0):
    times = pd.date_range(datetime.now() - timedelta(minutes=15*n), periods=n, freq="15min")
    data = {
        "close": [base_close] * n,
        "high": [base_close + 0.01] * n,
        "low": [base_close - 0.01] * n,
        "tick_volume": [5000] * n
    }
    df = pd.DataFrame(data, index=times)
    return df

class TestSpreadLogic(unittest.TestCase):
    def test_spread_calculation_stocks(self):
        df = make_df(n=300, base_close=30.0)

        mock_tick = MagicMock(ask=30.02, bid=30.00, last=30.01)

        with patch('utils.mt5.symbol_info', return_value=MagicMock(point=0.01)):
            with patch('utils.mt5.symbol_info_tick', return_value=mock_tick):
                with patch('utils.safe_copy_rates', return_value=df):
                    with patch('utils.get_atr', return_value=0.5):
                        with patch('utils.get_adx', return_value=25):
                            ind = utils.quick_indicators_custom("PETR4", 15)
                            self.assertIsInstance(ind, dict)
                            self.assertEqual(ind["spread_points"], 2)
                            self.assertAlmostEqual(ind["spread_nominal"], 0.02)
                            self.assertAlmostEqual(ind["spread_pct"], 0.0667, places=4)

    def test_spread_calculation_indices(self):
        df = make_df(n=300, base_close=130000.0)

        mock_tick = MagicMock(ask=130005, bid=130000, last=130001)

        with patch('utils.mt5.symbol_info', return_value=MagicMock(point=1.0)):
            with patch('utils.mt5.symbol_info_tick', return_value=mock_tick):
                with patch('utils.safe_copy_rates', return_value=df):
                    with patch('utils.get_atr', return_value=50):
                        with patch('utils.get_adx', return_value=25):
                            ind = utils.quick_indicators_custom("WIN$", 15)
                            self.assertIsInstance(ind, dict)
                            self.assertEqual(ind["spread_points"], 5)
                            self.assertAlmostEqual(ind["spread_nominal"], 5.0)

if __name__ == '__main__':
    unittest.main()
