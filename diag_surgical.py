import MetaTrader5 as mt5
import sys

def pf(*a): print(*a); sys.stdout.flush()

MT5_PATH = r"C:\MetaTrader 5 Terminal\terminal64.exe"

# 1. Tenta inicializar RAPIDAMENTE (sem path primeiro)
pf("--- Inicializando MT5 (sem path) ---")
if mt5.initialize():
    pf("✅ Sucesso (sem path)")
else:
    pf(f"❌ Falha (sem path): {mt5.last_error()}")
    pf(f"--- Inicializando MT5 (com path: {MT5_PATH}) ---")
    if mt5.initialize(path=MT5_PATH):
        pf("✅ Sucesso (com path)")
    else:
        pf(f"❌ Falha (com path): {mt5.last_error()}")
        sys.exit(1)

# 2. Informações do servidor
info = mt5.account_info()
term = mt5.terminal_info()
if info:
    pf(f"Servidor: {info.server}")
    pf(f"Broker  : {info.company}")
if term:
    pf(f"Conectado: {term.connected}")

# 3. Busca cirúrgica por WDO e WIN
for base in ["WDO", "WIN", "DOL", "IND"]:
    pf(f"\nBusca por *{base}*:")
    syms = mt5.symbols_get(group=f"*{base}*")
    if syms:
        for s in syms[:5]:
            pf(f"  - {s.name}")
    else:
        pf(f"  (nenhum encontrado para *{base}*)")

# 4. Tenta selecionar o genérico diretamente
for g in ["WDO$N", "WIN$N"]:
    sel = mt5.symbol_select(g, True)
    pf(f"\nSeleção de {g}: {sel}")
    if sel:
        inf = mt5.symbol_info(g)
        pf(f"  Visível: {inf.visible if inf else 'N/A'}")

mt5.shutdown()
