import MetaTrader5 as mt5
import sys
import os
from datetime import datetime

# Adiciona o diretório atual ao path
sys.path.append(os.getcwd())

import utils

def detailed_diag(symbol):
    if not mt5.initialize():
        print("Erro ao inicializar MT5")
        return

    print(f"\n===== Diagnóstico Detalhado: {symbol} =====")
    
    # Check what resolve_symbol sees
    s = symbol.upper()
    # Manual extraction to compare with utils.py
    base_manual = "".join([c for c in s.split("$")[0] if c.isalpha()])
    print(f"Base (Manual fix logic): {base_manual}")
    
    # Call the actual function
    final = utils.resolve_symbol(symbol)
    print(f"utils.resolve_symbol({symbol}) -> {final}")
    
    # Deep dive into get_contrato_atual
    res_atual = utils.get_contrato_atual(base_manual)
    print(f"utils.get_contrato_atual({base_manual}) -> {res_atual}")
    
    if res_atual:
        info = mt5.symbol_info(res_atual)
        if info:
            print(f"Propriedades de {res_atual}:")
            print(f"  Trade Mode: {info.trade_mode}")
        else:
            print(f"❌ Erro: {res_atual} não encontrado no MT5.")
    
    mt5.shutdown()

if __name__ == "__main__":
    detailed_diag("WIN$N")
    detailed_diag("WDO$N")
