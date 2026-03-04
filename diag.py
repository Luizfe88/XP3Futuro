"""
Diagnóstico MT5 - Mostra qual servidor está conectado
e lista TODOS os símbolos disponíveis.
"""
import MetaTrader5 as mt5

MT5_PATH = r"C:\MetaTrader 5 Terminal\terminal64.exe"

if not mt5.initialize(path=MT5_PATH):
    print(f"Erro ao inicializar MT5: {mt5.last_error()}")
    exit()

info     = mt5.account_info()
terminal = mt5.terminal_info()

print(f"\n{'='*55}")
print(f"  INFORMAÇÕES DA CONTA MT5")
print(f"{'='*55}")
print(f"  Servidor    : {info.server}")
print(f"  Corretora   : {info.company}")
print(f"  Login       : {info.login}")
print(f"  Nome        : {info.name}")
print(f"  Moeda       : {info.currency}")
print(f"  Terminal    : {terminal.name} v{terminal.build}")
print(f"{'='*55}\n")

all_syms = mt5.symbols_get() or []
print(f"  Total de símbolos disponíveis: {len(all_syms)}\n")

print(f"  Primeiros 60 símbolos (amostra):")
for s in all_syms[:60]:
    print(f"    {s.name:<25}  {s.description[:45] if s.description else ''}")

print(f"\n{'='*55}")
print(f"  Busca por prefixos de futuros B3:")
print(f"{'='*55}")
b3_prefixes = ["WIN", "IND", "WSP", "WDO", "DOL", "CCM", "BGI", "ICF", "BIT", "DI1"]
for pfx in b3_prefixes:
    matches = [s.name for s in all_syms if s.name.upper().startswith(pfx.upper())]
    label = ", ".join(matches[:8]) if matches else "(nenhum)"
    print(f"  {pfx:>5}: {label}")

mt5.shutdown()