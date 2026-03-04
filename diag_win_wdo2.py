import MetaTrader5 as mt5
import sys
import config

def print_flush(*args):
    print(*args)
    sys.stdout.flush()

def check():
    path = getattr(config, "MT5_TERMINAL_PATH", None)
    print_flush(f"🔌 Conectando ao MetaTrader 5 em {path}...")
    
    if not mt5.initialize():
        print_flush("Falha ao inicializar sem path, tentando com path...")
        if not mt5.initialize(path=path):
            print_flush("Falha ao inicializar MT5")
            return
            
    print_flush("Conectado.")
    
    # Testa os contratos diretamente
    test_symbols = ['WINJ26', 'WDOJ26', 'WIN$N', 'WDO$N', 'WIN@N', 'WDO@N']
    for s in test_symbols:
        info = mt5.symbol_info(s)
        if info:
            print_flush(f"✅ {s}: EXISTE. Selectable={info.selectable}, Visible={info.visible}")
        else:
            print_flush(f"❌ {s}: NÃO EXISTE ou INACESSÍVEL.")

    # Tenta usar o group parameter
    print_flush("\n--- Buscando pelo grupo *WIN* ---")
    win_group = mt5.symbols_get(group="*WIN*")
    if win_group:
        for s in win_group:
            print_flush(f"  Encontrado no grupo WIN: {s.name}")
    else:
        print_flush("  Nenhum encontrado no grupo *WIN*")

    print_flush("\n--- Buscando pelo grupo *WDO* ---")
    wdo_group = mt5.symbols_get(group="*WDO*")
    if wdo_group:
        for s in wdo_group:
            print_flush(f"  Encontrado no grupo WDO: {s.name}")
    else:
        print_flush("  Nenhum encontrado no grupo *WDO*")

    mt5.shutdown()

if __name__ == "__main__":
    check()
