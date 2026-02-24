import MetaTrader5 as mt5
from futures_core import FuturesDataManager

print("Inicializando MT5...")
if not mt5.initialize():
    print("Erro ao inicializar MT5")
    exit(1)

mgr = FuturesDataManager(mt5)

# Testa os bases que falharam
for base in ["IND", "DOL"]:
    print(f"\n{'='*60}")
    print(f"Testando base: {base}")
    print(f"{'='*60}")
    
    # Pega todos os símbolos
    all_symbols = mt5.symbols_get(group=f"*{base}*")
    print(f"Símbolos encontrados com *{base}*: {len(all_symbols) if all_symbols else 0}")
    
    if all_symbols:
        for s in all_symbols[:10]:  # Mostra os primeiros 10
            print(f"  - {s.name}")
    
    # Testa find_front_month
    print(f"\nChamando find_front_month('{base}')...")
    try:
        result = mgr.find_front_month(base)
        if result:
            print(f"✅ Resultado: {result}")
        else:
            print(f"❌ Resultado: None (não encontrado)")
    except Exception as e:
        print(f"❌ Erro: {e}")

mt5.shutdown()
print("\nMT5 desconectado.")
