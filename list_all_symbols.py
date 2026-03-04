import MetaTrader5 as mt5
import sys

def pf(*a): print(*a); sys.stdout.flush()

if not mt5.initialize():
    pf("❌ Falha na conexão MT5")
    sys.exit(1)

pf("✅ MT5 Conectado.")

# 1. Market Watch symbols
mw_syms = mt5.symbols_get()
pf(f"📊 Símbolos no Market Watch (Total {len(mw_syms)}):")
for s in mw_syms[:20]:
    pf(f"   - {s.name}")

# 2. Server symbols (top 50)
pf("\n📊 Símbolos no Servidor (Amostra dos primeiros 50):")
server_syms = mt5.symbols_get(group="*")
if server_syms:
    pf(f"   Total no servidor: {len(server_syms)}")
    # Ordena para ver se achamos WIN ou algo parecido
    names = sorted([s.name for s in server_syms])
    for n in names[:100]:
        if "WIN" in n.upper() or "WDO" in n.upper() or "$" in n:
             pf(f"   ⭐ {n}")
        else:
             # Só mostra os primeiros 20 genéricos se não for interessante
             pass
    
    # Busca específica sem case sensitive
    pf("\n🔍 Buscando por sub-strings específicas no servidor:")
    targets = ["WIN", "WDO", "IND", "DOL", "CCM", "BGI", "ICF"]
    for t in targets:
        matches = [n for n in names if t in n.upper()]
        pf(f"   - '{t}': {len(matches)} matches. Primeiros 5: {matches[:5]}")

mt5.shutdown()
