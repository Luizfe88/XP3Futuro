import pytest
import pandas as pd
import numpy as np
import sys
import os
from unittest.mock import MagicMock, patch

# Adiciona o diretório raiz ao sys.path para importar os módulos do bot
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import utils
import config

# --- Mocks ---
@pytest.fixture
def mock_mt5():
    with patch('utils.mt5') as mock:
        yield mock

@pytest.fixture
def sample_df():
    data = {
        'open': [10.0, 10.1, 10.2],
        'high': [10.2, 10.3, 10.4],
        'low': [9.8, 9.9, 10.0],
        'close': [10.1, 10.2, 10.3],
        'tick_volume': [100, 110, 120]
    }
    return pd.DataFrame(data)

# --- Testes de utilitários ---

def test_is_valid_dataframe_valid(sample_df):
    assert utils.is_valid_dataframe(sample_df) is True

def test_is_valid_dataframe_empty():
    assert utils.is_valid_dataframe(pd.DataFrame()) is False

def test_is_valid_dataframe_none():
    assert utils.is_valid_dataframe(None) is False

def test_is_valid_dataframe_min_rows(sample_df):
    assert utils.is_valid_dataframe(sample_df, min_rows=5) is False
    assert utils.is_valid_dataframe(sample_df, min_rows=3) is True

def test_calculate_signal_score_neutral():
    ind = {
        "rsi": 50,
        "adx": 20,
        "volume_ratio": 1.0,
        "momentum": 0.0,
        "ema_fast": 10,
        "ema_slow": 9,
        "macd": 0,
        "macd_signal": 0
    }
    score = utils.calculate_signal_score(ind)
    assert 90 <= score <= 100

def test_calculate_signal_score_strong_buy():
    ind = {
        "rsi": 50,
        "adx": 25,
        "volume_ratio": 1.5,
        "momentum": 0.02,
        "ema_fast": 10.5,
        "ema_slow": 10.0,
        "macd": 0.1,
        "macd_signal": 0.05,
        "stoch_k": 20
    }
    score = utils.calculate_signal_score(ind)
    assert score >= 60

# --- Testes de validação (Mocando MT5) ---

@patch('utils.mt5.positions_get')
@patch('utils.mt5.account_info')
def test_validate_subsetor_exposure_ok(mock_acc, mock_pos):
    # Simula conta com 100k e nenhuma posição
    mock_acc.return_value = MagicMock(equity=100000.0)
    mock_pos.return_value = []
    
    allowed, reason = utils.validate_subsetor_exposure("PETR4")
    assert allowed is True
    assert reason == "OK"

@patch('utils.mt5.positions_get')
@patch('utils.mt5.account_info')
def test_validate_subsetor_exposure_limit_exceeded(mock_acc, mock_pos):
    # Simula conta com 100k e 25k em PETR4 (Subsetor Petróleo)
    mock_acc.return_value = MagicMock(equity=100000.0)
    mock_pos.return_value = [
        MagicMock(symbol="PETR4", volume=1000, price_open=25.0)
    ]
    
    # Garantir que o mapa tenha o subsetor
    with patch.dict('config.SUBSETOR_MAP', {"PETR4": "Petróleo", "PETR3": "Petróleo"}):
        allowed, reason = utils.validate_subsetor_exposure("PETR3") # PETR3 também é Petróleo
        assert allowed is False
        assert "Exposição excessiva" in reason

if __name__ == "__main__":
    pytest.main()
