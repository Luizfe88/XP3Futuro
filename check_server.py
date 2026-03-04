import MetaTrader5 as mt5
import sys

def pf(*a): print(*a); sys.stdout.flush()

if not mt5.initialize():
    pf("❌ Falha na conexão MT5")
    sys.exit(1)

info = mt5.terminal_info()
pf(f"SERVER: {getattr(info, 'server', 'N/A')}")
pf(f"COMPANY: {getattr(info, 'company', 'N/A')}")
pf(f"ACC: {mt5.account_info().login if mt5.account_info() else 'N/A'}")

mt5.shutdown()
