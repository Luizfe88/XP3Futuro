import utils
import MetaTrader5 as mt5
from datetime import datetime

if mt5.initialize(path=r"C:\MetaTrader 5 Terminal\terminal64.exe"):
    bases = ["WIN", "IND", "WDO", "DOL", "WSP", "CCM", "BGI", "ICF"]
    print(f"{'Base':<8} | {'$N Ticker':<10} | {'Resolvido':<10} | {'Volume':<10}")
    print("-" * 45)
    for b in bases:
        target = f"{b}$N"
        res = utils.get_contrato_atual(target)
        vol = "N/A"
        if res:
            info = mt5.symbol_info(res)
            if info:
                vol = getattr(info, "volume", 0)
        print(f"{b:<8} | {target:<10} | {str(res):<10} | {vol:<10}")
    mt5.shutdown()
else:
    print("Erro MT5")
