import MetaTrader5 as mt5
import sys

try:
    import utils
    import config
    mt5.initialize(path=getattr(config, "MT5_TERMINAL_PATH", None))
    print(f"MT5 Inicializado: {mt5.terminal_info().connected}")
    
    bases = ["WIN", "IND", "WDO", "DOL", "WSP", "CCM", "BGI", "ICF", "DI1", "BIT", "T10"]
    print("RESOLUÇÃO DE ATIVOS (Contrato Atual):")
    for b in bases:
        c = utils.get_contrato_atual(b)
        print(f" {b.ljust(5)} -> {c}")
        if c:
            tick = mt5.symbol_info_tick(c)
            if tick:
                print(f"   Tick: {tick.last} / {tick.bid} / {tick.ask}")
            else:
                print(f"   [Sem dados de Tick para {c}]")

    print("\nRESOLUÇÃO DE SIMBOLOS DE DADOS PARA GENÉRICOS ($N):")
    for b in bases:
        sym = f"{b}$N"
        res = utils.resolve_symbol(sym)
        print(f" {sym.ljust(6)} -> {res}")
        
except Exception as e:
    print(f"Error: {e}")
finally:
    mt5.shutdown()
