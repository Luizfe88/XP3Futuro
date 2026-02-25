import MetaTrader5 as mt5
import re
from datetime import datetime, timedelta

print("Inicializando MT5...")
if not mt5.initialize():
    print("Erro ao inicializar MT5")
    exit(1)

# Testa IND e DOL manualmente
for base in ["IND", "DOL"]:
    print(f"\n{'='*70}")
    print(f"DEBUG: {base}")
    print(f"{'='*70}")
    
    # 1. Lista símbolos
    all_symbols = mt5.symbols_get(group=f"*{base}*")
    print(f"1. Símbolos encontrados: {len(all_symbols) if all_symbols else 0}")
    
    if not all_symbols:
        print("   Nenhum símbolo encontrado!")
        continue
    
    # 2. Mostra alguns
    for s in all_symbols[:5]:
        print(f"   - {s.name}")
    
    # 3. Testa regex
    pattern_str = f"{base}[FGHJKMNQUVXZ]\\d{{2}}"
    regex = re.compile(pattern_str)
    print(f"\n2. Regex: {pattern_str}")
    
    matches = []
    for s in all_symbols:
        if regex.search(s.name):
            matches.append(s.name)
    
    print(f"   Matches: {len(matches)}")
    for m in matches[:5]:
        print(f"   - {m}")
    
    # 4. Testa info do primeiro match
    if matches:
        test_symbol = matches[0]
        print(f"\n3. Testando info de: {test_symbol}")
        info = mt5.symbol_info(test_symbol)
        
        if info:
            print(f"   ✅ Info obtida")
            print(f"   - visible: {info.visible}")
            print(f"   - expiration_time: {info.expiration_time}")
            
            # Testa conversão de timestamp
            if info.expiration_time and info.expiration_time > 0:
                try:
                    exp_date = datetime.fromtimestamp(info.expiration_time)
                    print(f"   - expiry_date: {exp_date}")
                    days_to_exp = (exp_date - datetime.now()).days
                    print(f"   - days_to_expiry: {days_to_exp}")
                except Exception as e:
                    print(f"   ❌ Erro ao converter timestamp: {e}")
            else:
                print(f"   - expiration_time é 0 ou None")
            
            # Open Interest
            oi = getattr(info, "session_open_interest", None)
            if oi in (None, 0):
                oi = getattr(info, "open_interest", None)
            print(f"   - OI: {oi}")
        else:
            print(f"   ❌ Não foi possível obter info")

mt5.shutdown()
print("\n✅ MT5 desconectado.")
