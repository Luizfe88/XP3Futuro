import sys
import os
import unittest
from unittest.mock import MagicMock

# Adiciona o diretório atual ao path para importar os módulos
sys.path.append(os.getcwd())

import config
import utils

class TestSignalScore(unittest.TestCase):
    def setUp(self):
        # Mock de indicadores básicos
        self.ind = {
            "symbol": "WIN$N",
            "rsi": 60,
            "adx": 12, # Baixo para o padrão antigo (18-20)
            "volume_ratio": 0.5,
            "ema_fast": 100.5,
            "ema_slow": 100.0, # Tendência de alta
            "macd": 0.1,
            "macd_signal": 0.05,
            "atr_real": 0.1,
            "close": 100.2
        }

    def test_conservative_rejection(self):
        config.DEFAULT_OPERATION_MODE = "CONSERVADOR"
        score = utils.calculate_signal_score(self.ind)
        # Em modo conservador, adx_min = 20. Com ADX=12, deve falhar no ADX_OK
        # O score deve ser menor que em outros modos
        print(f"Score Conservador (ADX 12): {score}")
        self.assertTrue(score < 60) # Valor aproximado baseado na lógica

    def test_balanced_adx_logic(self):
        config.DEFAULT_OPERATION_MODE = "BALANCED"
        # Em balanced, adx_min = 15. Com ADX=12, ainda não ganha o bônus de ADX_OK
        score = utils.calculate_signal_score(self.ind)
        print(f"Score Balanced (ADX 12): {score}")
        
        # Testando com ADX 16
        self.ind["adx"] = 16
        score_ok = utils.calculate_signal_score(self.ind)
        print(f"Score Balanced (ADX 16): {score_ok}")
        self.assertTrue(score_ok > score)

    def test_aggressive_adx_logic(self):
        config.DEFAULT_OPERATION_MODE = "AGRESSIVO"
        self.ind["adx"] = 12
        # Em agressivo, adx_min = 10. Com ADX=12, deve passar no ADX_OK
        score = utils.calculate_signal_score(self.ind)
        print(f"Score Agressivo (ADX 12): {score}")
        self.assertTrue(score >= 40) # Deve ganhar pontos de ADX_OK e VOLUME_OK etc.

if __name__ == "__main__":
    unittest.main()
