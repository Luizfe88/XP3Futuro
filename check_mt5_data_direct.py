import MetaTrader5 as mt5
import pandas as pd
import sys
from datetime import datetime

def pf(*a): print(*a); sys.stdout.flush()

if not mt5.initialize():
    pf("❌ MT5 Init failed")
    sys.exit(1)

pf("✅ MT5 Connected")

test_symbols = ["WDO$N", "WDOH26", "WDOJ26", "WIN$N", "WINJ26", "BGI$N", "BGIH26", "ICF$N", "ICFH26"]

for s in test_symbols:
    pf(f"\n--- Probando: {s} ---")
    select = mt5.symbol_select(s, True)
    pf(f"   Select: {select}")
    info = mt5.symbol_info(s)
    if info:
        pf(f"   Visible: {info.visible}, Select: {info.select}")
        pf(f"   Expiration: {datetime.fromtimestamp(info.expiration_time) if info.expiration_time else 'N/A'}")
        
        # Tenta pegar 10 candles
        rates = mt5.copy_rates_from_pos(s, mt5.TIMEFRAME_M5, 0, 10)
        if rates is not None and len(rates) > 0:
            df = pd.DataFrame(rates)
            pf(f"   ✅ RETORNOU {len(df)} candles do MT5.")
            pf(f"   Primeiro candle tempo: {datetime.fromtimestamp(df.iloc[0]['time'])}")
        else:
            pf(f"   ❌ SEM DADOS do MT5 (None ou zero)")
    else:
        pf(f"   ❌ SÍMBOLO NÃO ENCONTRADO no MT5")

# Verifica busca por grupo de novo pra ver se *WDO* falha
pf("\n--- Teste de busca por grupo *WDO* ---")
res = mt5.symbols_get(group="*WDO*")
if res:
    pf(f"   Encontrados {len(res)} símbolos:")
    for i, sym in enumerate(res[:10]):
        pf(f"      {sym.name}")
else:
    pf("   ❌ symbols_get(group='*WDO*') não retornou nada!")

mt5.shutdown()
