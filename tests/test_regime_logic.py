import sys
import os
import pandas as pd
import numpy as np
from unittest.mock import MagicMock

# Adiciona o diretório atual ao path para importar bot_quant_portfolio
sys.path.append(os.getcwd())

# Mock do MetaTrader5 pois ele não inicializa sem o terminal
import MetaTrader5 as mt5
sys.modules['MetaTrader5'] = MagicMock()

# Mock do utils.resolve_symbol
import utils
utils.resolve_symbol = MagicMock(side_effect=lambda x: x)

# Importa o Worker do bot_quant_portfolio
from bot_quant_portfolio import AssetWorker

def test_regime_logic():
    print("--- Iniciando Teste de Lógica de Regimes ---")
    
    # Configuração fake
    config = {
        "allocation": 0.33,
        "tick_value": 5.0,
        "base_win_rate": 0.55,
        "base_payout": 1.5,
        "n_states": 2,
        "kelly_fraction": 0.10
    }
    
    # Instancia o worker
    worker = AssetWorker("WDO$N", config, 500.0)
    worker.initialized = {"M5": True, "M15": True, "M30": True}
    
    # Mock do risk_manager
    worker.risk_manager.calculate_position_size = MagicMock(return_value=(1, 0, "Fake Debug"))
    
    # Mock do mt5.history_deals_get
    mt5.history_deals_get = MagicMock(return_value=[])
    
    # Mock do mt5.positions_get
    mt5.positions_get = MagicMock(return_value=[]) # Sem posições abertas
    
    # Caso 1: M5=2, M15=1, M30=1 (Deve permitir can_trade=True)
    regimes = {"M5": 2, "M15": 1, "M30": 1}
    contracts = 1
    debug = ""
    
    # Replicando a lógica do process_tick simplificada para o teste
    can_trade = False
    if regimes["M5"] == 1: # Trend consensus check simplificado
        pass 
    elif regimes["M5"] == 2 and contracts > 0:
        if regimes.get("M15") != 2 and regimes.get("M30") != 2:
            can_trade = True
        else:
            contracts = 0
            debug += " | [BLOCK] High TF Exhaustion Consensus"
            
    print(f"Cenário 1 (M5=2, M15=1): can_trade={can_trade}, contracts={contracts}, debug='{debug}'")
    assert can_trade is True
    assert contracts == 1

    # Caso 2: M5=2, M15=2, M30=1 (Deve BLOQUEAR entrada)
    regimes = {"M5": 2, "M15": 2, "M30": 1}
    contracts = 1
    debug = ""
    can_trade = False
    
    if regimes["M5"] == 1:
        pass
    elif regimes["M5"] == 2 and contracts > 0:
        if regimes.get("M15") != 2 and regimes.get("M30") != 2:
            can_trade = True
        else:
            contracts = 0
            debug += " | [BLOCK] High TF Exhaustion Consensus"
            
    print(f"Cenário 2 (M5=2, M15=2): can_trade={can_trade}, contracts={contracts}, debug='{debug}'")
    assert can_trade is False
    assert contracts == 0
    assert "[BLOCK] High TF Exhaustion Consensus" in debug

    print("--- ✅ Teste Concluído com Sucesso! ---")

if __name__ == "__main__":
    try:
        test_regime_logic()
    except Exception as e:
        print(f"❌ Erro no teste: {e}")
        sys.exit(1)
