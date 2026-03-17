import MetaTrader5 as mt5
import sys
import os
from datetime import datetime

# Adiciona o diretório atual ao path
sys.path.append(os.getcwd())

import utils
import config

def detailed_diag(symbol):
    if not mt5.initialize():
        print("Erro ao inicializar MT5")
        return

    print(f"\n===== Diagnóstico Detalhado: {symbol} =====")
    
    # Step 1: Base extraction
    s = (symbol or "").upper().strip()
    base = "".join([c for c in s.replace("$", "").replace("N", "") if c.isalpha()])
    print(f"Base extraída: {base}")
    
    # Step 2: get_contrato_atual
    print("\n--- Testando get_contrato_atual(base) ---")
    res_atual = utils.get_contrato_atual(base)
    print(f"get_contrato_atual({base}) retornou: {res_atual}")
    
    # Step 3: calculate_current_b3_contract
    print("\n--- Testando calculate_current_b3_contract(base) ---")
    res_math = utils.calculate_current_b3_contract(base)
    print(f"calculate_current_b3_contract({base}) retornou: {res_math}")
    
    # Step 4: get_futures_candidates
    print("\n--- Testando get_futures_candidates(base, ignore_generic=True) ---")
    cands = utils.get_futures_candidates(base, ignore_generic=True)
    print(f"Candidatos encontrados ({len(cands)}):")
    for c in cands:
        print(f"  - {c['symbol']} (Volume: {c.get('volume', 0)})")
        
    # Step 5: Full resolve_symbol
    print("\n--- Testando resolve_symbol(symbol) ---")
    final = utils.resolve_symbol(symbol)
    print(f"resolve_symbol({symbol}) -> {final}")
    
    # Step 6: Symbol Info on Final
    info = mt5.symbol_info(final)
    if info:
        print(f"\nPropriedades de {final}:")
        print(f"  Visible: {info.visible}")
        print(f"  Trade Mode: {info.trade_mode}")
        print(f"  Trade Mode Text: {info.trade_mode}") # Mode mapping defined in previous script
    else:
        print(f"\n❌ Erro: {final} não encontrado no MT5.")

    mt5.shutdown()

if __name__ == "__main__":
    detailed_diag("WIN$N")
    detailed_diag("WDO$N")
