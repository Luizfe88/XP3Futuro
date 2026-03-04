"""
Script MÍNIMO - sem importar utils.
Testa diretamente quais símbolos o MT5 conhece para cada base de futuro.
"""
import MetaTrader5 as mt5
import re, sys

print("Conectando ao MT5...", flush=True)
ok = mt5.initialize()
print(f"Conectado: {ok}  |  Erro: {mt5.last_error()}", flush=True)
if not ok:
    sys.exit(1)

bases = ['WIN', 'WDO', 'IND', 'DOL', 'ICF', 'CCM', 'BIT', 'WSP', 'DI1', 'SFI']

print("\n--- Buscando por mask ---", flush=True)
for base in bases:
    syms = mt5.symbols_get(f"{base}*") or []
    names = [s.name for s in syms[:10]]
    print(f"  {base}*  ->  {names}", flush=True)

print("\n--- Futuros no Market Watch (todos) ---", flush=True)
all_syms = mt5.symbols_get() or []
futures = []
for s in all_syms:
    if re.match(r'^[A-Z]{2,6}[FGHJKMNQUVXZ]\d{2}$', s.name):
        futures.append(s.name)

futures.sort()
print(f"Total encontrado: {len(futures)}", flush=True)
for f in futures:
    print(f"  {f}", flush=True)

mt5.shutdown()
print("Pronto.", flush=True)
