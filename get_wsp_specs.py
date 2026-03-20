import MetaTrader5 as mt5
import json

def get_symbol_specs(symbol):
    if not mt5.initialize():
        return {"error": "Failed to initialize MT5"}
    
    mt5.symbol_select(symbol, True)
    info = mt5.symbol_info(symbol)
    if info is None:
        mt5.shutdown()
        return {"error": f"Symbol {symbol} not found"}
    
    specs = {
        "symbol": info.name,
        "tick_size": info.trade_tick_size,
        "tick_value": info.trade_tick_value,
        "point_value": info.trade_tick_value / info.trade_tick_size if info.trade_tick_size > 0 else 0,
        "margin": info.margin_initial,
        "digits": info.digits
    }
    mt5.shutdown()
    return specs

if __name__ == "__main__":
    print(json.dumps(get_symbol_specs("WSP$N")))
