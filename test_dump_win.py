import MetaTrader5 as mt5
import sys

def pf(*a): print(*a); sys.stdout.flush()

mt5.initialize()
syms = mt5.symbols_get(group="*WIN*")
if syms:
    for s in syms:
        pf(f"NOME ENCONTRADO: '{s.name}'")
else:
    pf("Nenhum símbolo retornado de symbols_get('*WIN*')")
mt5.shutdown()
