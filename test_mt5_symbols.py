"""
Diagnóstico de símbolos disponíveis no MT5 Market Watch.
Lista tudo que aparece para cada prefixo de futuro.
"""
import MetaTrader5 as mt5
import sys

mt5.initialize()

bases = ['WIN', 'WDO', 'IND', 'DOL', 'ICF', 'CCM', 'BIT', 'WSP', 'BGI', 'DI1', 'SFI', 'DI']

print("=" * 70)
print("DIAGNÓSTICO: Símbolos disponíveis no MT5 para cada base")
print("=" * 70)

for base in bases:
    # Busca sem mask (todos os símbolos com prefixo)
    all_syms = mt5.symbols_get() or []
    matches = [s.name for s in all_syms if s.name.startswith(base)]

    # Busca com mask pattern
    masked = mt5.symbols_get(f"{base}*") or []
    by_mask = [s.name for s in masked]

    print(f"\n📌 Base: {base}")
    print(f"   Por s.name.startswith: {matches[:8]}")
    print(f"   Por symbols_get('{base}*'): {by_mask[:8]}")

print("\n" + "=" * 70)
print("Lista completa de futuros B3 no Market Watch:")
all_syms = mt5.symbols_get() or []
futures = []
import re
for s in all_syms:
    if re.match(r'^[A-Z]{2,6}[FGHJKMNQUVXZ]\d{2}$', s.name):
        futures.append(s.name)

futures.sort()
for f in futures:
    info = mt5.symbol_info(f)
    visible = getattr(info, 'visible', False) if info else False
    print(f"  {'✅' if visible else '⬜'} {f}")

print(f"\nTotal de contratos futuros encontrados: {len(futures)}")
mt5.shutdown()
