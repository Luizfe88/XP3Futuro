import MetaTrader5 as mt5
import sys
import os

# Adiciona o diretório atual ao path
sys.path.append(os.getcwd())

import utils

def diagnose_symbols(symbols):
    if not mt5.initialize():
        print("Erro ao inicializar MT5")
        return

    for sym in symbols:
        print(f"\n--- Analisando: {sym} ---")
        resolved = utils.resolve_symbol(sym)
        print(f"Resolvido para: {resolved}")
        
        info = mt5.symbol_info(resolved)
        if info is None:
            print(f"❌ Erro: Não foi possível obter symbol_info para {resolved}")
            continue
            
        print(f"Visible: {info.visible}")
        print(f"Trade Mode: {info.trade_mode}")
        # Modes: 0 - Disabled, 1 - Long Only, 2 - Short Only, 3 - Close Only, 4 - Full
        trade_modes = {
            0: "SYMBOL_TRADE_MODE_DISABLED",
            1: "SYMBOL_TRADE_MODE_LONGONLY",
            2: "SYMBOL_TRADE_MODE_SHORTONLY",
            3: "SYMBOL_TRADE_MODE_CLOSEONLY",
            4: "SYMBOL_TRADE_MODE_FULL"
        }
        print(f"Mode Text: {trade_modes.get(info.trade_mode, 'UNKNOWN')}")
        
        tick = mt5.symbol_info_tick(resolved)
        if tick:
            print(f"Ask: {tick.ask}, Bid: {tick.bid}")
        else:
            print("❌ Erro: Não foi possível obter o tick atual.")

    mt5.shutdown()

if __name__ == "__main__":
    diagnose_symbols(["WIN$N", "WDO$N"])
