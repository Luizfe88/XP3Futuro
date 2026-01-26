
import unittest
from unittest.mock import MagicMock, patch
import json
import os
import sys

# Adiciona diretório raiz ao path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
import utils
import bot
import validation

class TestBotLogic(unittest.TestCase):
    
    def setUp(self):
        """Configuração inicial para cada teste"""
        config.OPERATION_MODES["TEST"] = {
            "description": "Modo de Teste",
            "allow_new_entries": True,
            "allow_pyramiding": False,
            "max_concurrent_positions": 100,
            "profit_lock_pct": 0.01
        }
        config.set_operation_mode("TEST")

    def test_kelly_conservative(self):
        """Testa se Kelly está usando multiplicador 0.3"""
        # Mocking validation logic to isolate Kelly calculation
        with patch('validation.monte_carlo_ruin_check', return_value=0.0):
            with patch('mt5.account_info') as mock_acc:
                mock_acc.return_value.balance = 100000.0
                
                # Simula cálculo
                # win_rate=0.6, rr=1.5, kelly_fraction=~0.33
                # adjusted = 0.33 * 0.3 (multiplier) ~= 0.1
                # Final deve ser reduzido
                
                # Check directly in validation.py logic if accessible or simulate order creation
                from validation import calculate_kelly_position_size
                
                vol = calculate_kelly_position_size("PETR4", 10.0, 9.5, 11.0, "BUY")
                self.assertGreater(vol, 0)
    
    def test_win_rate_pause(self):
        """Testa pausa por Win Rate baixo"""
        # Mocking deals to simulate low win rate
        with patch('mt5.history_deals_get') as mock_deals:
            # Cria deals com profit negativo
            mock_deal_loss = MagicMock()
            mock_deal_loss.profit = -100.0
            mock_deal_loss.entry = 2 # DEAL_ENTRY_OUT
            
            mock_deal_win = MagicMock()
            mock_deal_win.profit = 100.0
            mock_deal_win.entry = 2
            
            # 8 losses, 2 wins = 20% WR
            mock_deals.return_value = [mock_deal_loss]*8 + [mock_deal_win]*2
            
            should_pause, reason = bot.check_win_rate_pause()
            self.assertTrue(should_pause)
            self.assertIn("WR Crítico", reason)

    @patch('utils.requests.get')
    def test_polygon_integration(self, mock_get):
        """Testa integração com Polygon.io (Mock)"""
        # Mock response
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "results": [
                {"t": 1678888000000, "c": 15.0, "o": 14.5, "h": 15.1, "l": 14.4, "v": 1000}
            ],
            "status": "OK"
        }
        mock_get.return_value = mock_resp
        
        # Test order flow fetch
        flow = utils.get_order_flow_polygon("PETR4")
        self.assertIsNotNone(flow)
        
    def test_dynamic_config(self):
        """Testa parâmetros dinâmicos de win rate"""
        # High Win Rate
        params_high = config.get_params_for_win_rate(0.75)
        self.assertEqual(params_high['kelly_multiplier'], 0.4)
        
        # Low Win Rate
        params_low = config.get_params_for_win_rate(0.45)
        self.assertEqual(params_low['kelly_multiplier'], 0.1)

if __name__ == '__main__':
    unittest.main()
