import MetaTrader5 as mt5

def check():
    if not mt5.initialize():
        print("Falha ao inicializar MT5")
        return
    
    print("\n--- Buscando todos os símbolos ---")
    all_symbols = mt5.symbols_get()
    if not all_symbols:
        print("Nenhum símbolo retornado por symbols_get()!")
    else:
        print(f"Total de símbolos na corretora: {len(all_symbols)}")
        win_list = [s.name for s in all_symbols if "WIN" in s.name.upper()]
        wdo_list = [s.name for s in all_symbols if "WDO" in s.name.upper()]
        
        print("\n--- Símbolos contendo WIN ---")
        for s in win_list:
            print(s)
            
        print("\n--- Símbolos contendo WDO ---")
        for s in wdo_list:
            print(s)

    mt5.shutdown()

if __name__ == "__main__":
    check()
