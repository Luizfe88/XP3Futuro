import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# Adiciona o diretório atual ao path para importar utils
sys.path.insert(0, os.path.abspath(os.curdir))

# Mock MetaTrader5 before importing utils if it's used at module level
sys.modules['MetaTrader5'] = MagicMock()
import MetaTrader5 as mt5

# Mock other imports that might fail
sys.modules['config'] = MagicMock()
sys.modules['database'] = MagicMock()
sys.modules['news_filter'] = MagicMock()
sys.modules['telegram_handler'] = MagicMock()
sys.modules['adaptive_system'] = MagicMock()
sys.modules['hedging'] = MagicMock()
sys.modules['ml_optimizer'] = MagicMock()
sys.modules['daily_analysis_logger'] = MagicMock()
sys.modules['daily_logger'] = MagicMock()
sys.modules['validation'] = MagicMock()

import utils

class TestCorrelationFix(unittest.TestCase):
    @patch('utils.mt5.copy_rates_from_pos')
    @patch('utils.pd.DataFrame')
    def test_update_correlations_mixed_lengths(self, mock_df, mock_copy_rates):
        # Setup mock: SYM1=50, SYM2=49, SYM3=50
        def side_effect(sym, timeframe, start, count):
            if sym == "SYM1":
                return [{"close": i} for i in range(50)]
            if sym == "SYM2":
                return [{"close": i} for i in range(49)]
            if sym == "SYM3":
                return [{"close": i} for i in range(50)]
            return None
        
        mock_copy_rates.side_effect = side_effect
        
        # Call the function
        symbols = ["SYM1", "SYM2", "SYM3"]
        utils.update_correlations(symbols)
        
        # Verify that DataFrame was called with SYM1 and SYM3
        self.assertTrue(mock_df.called, "pd.DataFrame should have been called")
        args, kwargs = mock_df.call_args
        data_arg = args[0]
        
        self.assertIn("SYM1", data_arg)
        self.assertNotIn("SYM2", data_arg)
        self.assertIn("SYM3", data_arg)
        self.assertEqual(len(data_arg["SYM1"]), 50)
        self.assertEqual(len(data_arg["SYM3"]), 50)
        print("✅ Verification passed: SYM2 (49 bars) was correctly filtered, and correlation continued with SYM1 and SYM3.")

if __name__ == '__main__':
    unittest.main()
