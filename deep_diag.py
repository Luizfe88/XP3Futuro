import MetaTrader5 as mt5
import sys

def pf(*a): print(*a); sys.stdout.flush()

if not mt5.initialize():
    pf("❌ Falha na conexão MT5")
    sys.exit(1)

pf("✅ MT5 Conectado.")
info = mt5.terminal_info()
if info:
    pf(f"🖥️ Terminal Server: {getattr(info, 'server', 'N/A')}")
    pf(f"🏦 Terminal Company: {getattr(info, 'company', 'N/A')}")
    pf(f"✅ Conectado: {getattr(info, 'connected', 'N/A')}")

total = mt5.symbols_total()
pf(f"📊 Total de símbolos no terminal: {total}")

# Busca padrões comuns
patterns = ["*WIN*", "WIN*", "*WDO*", "WDO*", "*IND*", "*DOL*", "*BIT*", "*IBOV*", "*PETR*"]
for p in patterns:
    res = mt5.symbols_get(group=p)
    count = len(res) if res else 0
    pf(f"🔍 Busca '{p}': {count} resultados")
    if res:
        for i, s in enumerate(res[:5]):
            pf(f"   [{i}] {s.name}")

# Tenta mt5.symbol_info para os que o usuário quer
wanted = ['WIN$N','IND$N','WSP$N','WDO$N','DOL$N','CCM$N','BGI$N','ICF$N','BIT$N','DI1$N']
pf("\n🔍 Verificando os símbolos solicitados pelo usuário:")
for w in wanted:
    inf = mt5.symbol_info(w)
    if inf:
        pf(f"   ✅ {w} EXISTE (Visível: {inf.visible}, Select: {mt5.symbol_select(w, True)})")
    else:
        pf(f"   ❌ {w} NÃO EXISTE")

mt5.shutdown()
