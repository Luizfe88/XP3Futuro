import MetaTrader5 as mt5
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import utils

def adx_safe(df):
    v = utils.get_adx(df)
    if v is None and df is not None:
        try:
            v = utils.get_adx(df.tail(60))
        except Exception:
            v = None
    if v is None and df is not None:
        try:
            v = utils.get_adx(df, period=8)
        except Exception:
            v = None
    return v

def run_for_symbol(symbol):
    try:
        mt5.symbol_select(symbol, True)
    except Exception:
        pass
    for tf, name in [(mt5.TIMEFRAME_M5, "M5"), (mt5.TIMEFRAME_M15, "M15")]:
        df = utils.safe_copy_rates(symbol, tf, 200)
        bars = 0 if df is None else len(df)
        val = adx_safe(df)
        v = 0.0 if val is None else float(val)
        print(f"{symbol} {name} ADX={v:.2f} barras={bars}")

def main():
    try:
        mt5.initialize()
    except Exception:
        pass
    for s in ["WING26", "INDG26", "WDOH26", "DOLH26"]:
        run_for_symbol(s)

if __name__ == "__main__":
    main()
