import unittest
from investment_allocation import allocate_assets

class TestAllocation(unittest.TestCase):
    def setUp(self):
        self.cfg = {
            "category": "BLUE CHIP",
            "weight": 0.08,
            "ema_short": 21,
            "ema_long": 90,
            "rsi_low": 34,
            "rsi_high": 60,
            "adx_threshold": 28.0,
            "mom_min": 0.0,
            "sl_atr_multiplier": 3.1,
            "tp_mult": 3.0
        }
        self.assets = [
            {"symbol": "VALE3", "ema_short": 105.0, "ema_long": 100.0, "rsi": 50.0, "adx": 32.0, "mom": 0.02, "sl_atr": 3.0},
            {"symbol": "PETR4", "ema_short": 90.0, "ema_long": 100.0, "rsi": 45.0, "adx": 20.0, "mom": -0.01, "sl_atr": 3.5},
            {"symbol": "ITUB4", "ema_short": 110.0, "ema_long": 100.0, "rsi": 40.0, "adx": 30.0, "mom": 0.01, "sl_atr": 2.8}
        ]

    def test_validation_rules(self):
        bad = dict(self.cfg)
        bad["ema_short"] = 100
        bad["ema_long"] = 90
        with self.assertRaises(ValueError):
            allocate_assets(self.assets, bad)
        bad2 = dict(self.cfg)
        bad2["rsi_low"] = 70
        bad2["rsi_high"] = 60
        with self.assertRaises(ValueError):
            allocate_assets(self.assets, bad2)
        bad3 = dict(self.cfg)
        bad3["sl_atr_multiplier"] = 0
        with self.assertRaises(ValueError):
            allocate_assets(self.assets, bad3)

    def test_allocation_sum_and_qualification(self):
        res = allocate_assets(self.assets, self.cfg)
        total = res["total_percent"]
        self.assertTrue(0.0 < total <= 100.0)
        syms = [a["symbol"] for a in res["allocations"]]
        self.assertIn("VALE3", syms)
        self.assertIn("ITUB4", syms)
        self.assertNotIn("PETR4", syms)

if __name__ == "__main__":
    unittest.main()
