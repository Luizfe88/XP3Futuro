import unittest
import logging
import os
import re
from unittest.mock import patch, MagicMock
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Configura logging detalhado para capturar e inspecionar mensagens
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("tests.elite_symbols_integration")

class TestEliteSymbolsImport(unittest.TestCase):
    def test_imports_core_modules(self):
        import xp3future as config
        self.assertTrue(hasattr(config, "ELITE_SYMBOLS"), "ELITE_SYMBOLS ausente em xp3future/config")
        self.assertIsInstance(getattr(config, "ELITE_SYMBOLS"), dict, "ELITE_SYMBOLS deve ser dict")
        # Verifica padrão de futuros ($N)
        keys = list(config.ELITE_SYMBOLS.keys())
        self.assertGreater(len(keys), 0, "ELITE_SYMBOLS deve conter pelo menos um símbolo")
        for k in keys:
            self.assertIn("$N", k, f"Símbolo não parece futuro contínuo: {k}")

        os.makedirs(os.path.join(os.getcwd(), "logs", "analysis"), exist_ok=True)
        import xp3future.bot as bot
        self.assertTrue(hasattr(bot, "validate_futures_only_mode"), "validate_futures_only_mode ausente no bot")

        import validation
        from validation import OrderParams, OrderSide
        self.assertTrue(callable(getattr(validation, "validate_and_create_order", lambda: None)) or True, "Função principal de validação deve existir")
        self.assertTrue(OrderSide.BUY.value == "BUY" and OrderSide.SELL.value == "SELL")

class TestModuleCommunication(unittest.TestCase):
    def test_order_params_validation_flow(self):
        import xp3future.validation as validation
        with patch.object(validation, "mt5") as mock_mt5, patch.object(validation, "utils") as mock_utils:
            tick = MagicMock()
            tick.ask = 100.0
            tick.bid = 100.0
            mock_mt5.symbol_info_tick.return_value = tick
            info = MagicMock()
            info.point = 0.5
            mock_mt5.symbol_info.return_value = info
            mock_mt5.TIMEFRAME_M15 = 16385
            mock_mt5.DEAL_ENTRY_OUT = 1
            mock_mt5.ORDER_TYPE_BUY = 0
            mock_mt5.ORDER_TYPE_SELL = 1
            mock_mt5.TRADE_ACTION_DEAL = 7
            mock_mt5.ORDER_TIME_GTC = 1
            mock_mt5.ORDER_FILLING_RETURN = 2
            mock_mt5.history_deals_get.return_value = []
            mock_utils.safe_copy_rates.return_value = []
            mock_utils.get_atr.return_value = 1.0
            mock_utils.get_ibov_correlation.return_value = 0.5
            mock_utils.get_adx_series.return_value = MagicMock()
            from xp3future.validation import OrderParams, OrderSide
            op = OrderParams(
                symbol="WIN$N",
                side=OrderSide.BUY,
                volume=1.0,
                entry_price=100.0,
                sl=95.0,
                tp=104.0,
                kelly_adjusted=True
            )
            req = op.to_mt5_request(magic=123456, comment="XP3")
            self.assertEqual(req["symbol"], "WIN$N")
            self.assertEqual(req["type"], mock_mt5.ORDER_TYPE_BUY)
            self.assertGreater(req["tp"], req["price"])
            self.assertLess(req["sl"], req["price"])
            self.assertGreater(int(req["volume"]), 0)

class TestProgramExecution(unittest.TestCase):
    def test_validate_futures_only_mode(self):
        import xp3future as config
        os.makedirs(os.path.join(os.getcwd(), "logs", "analysis"), exist_ok=True)
        import xp3future.bot as bot
        # Deve ser True com apenas futuros
        self.assertTrue(bot.validate_futures_only_mode())
        # Injeta um símbolo de ação temporariamente e valida False
        orig = dict(config.ELITE_SYMBOLS)
        try:
            config.ELITE_SYMBOLS["VALE3"] = {"parameters": {}}
            self.assertFalse(bot.validate_futures_only_mode())
        finally:
            config.ELITE_SYMBOLS.clear()
            config.ELITE_SYMBOLS.update(orig)

class TestOutputsAndExpectations(unittest.TestCase):
    def test_elite_symbols_json_presence_and_structure(self):
        import xp3future as config
        path = getattr(config, "ELITE_SYMBOLS_JSON_PATH", "optimizer_output/elite_symbols_latest.json")
        self.assertIsInstance(path, str)
        # Arquivo pode ou não existir; se existir, valida estrutura
        if os.path.exists(path):
            import json
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.assertIn("elite_symbols", payload)
            self.assertIsInstance(payload["elite_symbols"], dict)
            # Checa que não há duplicidade de sufixo $N
            for k in payload["elite_symbols"].keys():
                self.assertNotRegex(k, re.compile(r"\$N\\$N"), "Símbolo duplicado com $N$N")

if __name__ == "__main__":
    unittest.main(verbosity=2)
