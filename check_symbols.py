import MetaTrader5 as mt5
import sys

def pf(*a): print(*a); sys.stdout.flush()

pf('Conectando...')
if not mt5.initialize():
    pf('Falha:', mt5.last_error())
    sys.exit(1)

pf('Conectado.')
tests = ['WIN$N', 'WDO$N', 'DI1$N', 'WINN', 'WDON', 'WIN@N', 'WDO@N', 'WINJ26', 'WDOM26']
for t in tests:
    sel = mt5.symbol_select(t, True)
    info = mt5.symbol_info(t) if sel else None
    pf(f'{t}: select={sel}, info={info is not None}')

mt5.shutdown()
