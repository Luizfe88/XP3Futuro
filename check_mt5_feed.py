import MetaTrader5 as mt5
import config
import utils


def _fmt(v):
    try:
        if v is None:
            return "None"
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)
    except Exception:
        return str(v)


def main():
    lines = []
    try:
        ok = mt5.initialize(path=getattr(config, "MT5_TERMINAL_PATH", None))
        lines.append(f"MT5.initialize: {ok}")
        term = mt5.terminal_info()
        lines.append(f"terminal.connected: {getattr(term, 'connected', None)}")
        lines.append(f"terminal.trade_allowed: {getattr(term, 'trade_allowed', None)}")

        targets = ["WDOH26", "DOLH26", "WSPH26", "BGIH26", "WING26", "INDG26"]
        for s in targets:
            try:
                base = utils._futures_base_from_symbol(s)
                current = utils.get_current_contract(base)
                data_sym = utils.resolve_indicator_symbol(s)
                trade_sym = utils.resolve_trade_symbol(s)

                tick = mt5.symbol_info_tick(trade_sym)
                bid = getattr(tick, "bid", 0.0) if tick else 0.0
                ask = getattr(tick, "ask", 0.0) if tick else 0.0

                df = utils.safe_copy_rates(data_sym, mt5.TIMEFRAME_M15, 300)
                rows = 0 if df is None else len(df)
                last_close = None
                if df is not None and len(df) > 0 and "close" in df.columns:
                    try:
                        last_close = float(df["close"].iloc[-1])
                    except Exception:
                        last_close = None

                ind = utils.quick_indicators_custom(s, mt5.TIMEFRAME_M15, df=None, params={})
                lines.append(
                    f"{s} | base={base} | current={current} | data={data_sym} | trade={trade_sym} | "
                    f"tick={_fmt(bid)}/{_fmt(ask)} | rows={rows} | last_close={_fmt(last_close)} | "
                    f"rsi={_fmt(ind.get('rsi'))} | adx={_fmt(ind.get('adx'))} | volx={_fmt(ind.get('volume_ratio'))} | err={ind.get('error')}"
                )
            except Exception as e:
                lines.append(f"{s} | ERROR: {e}")
    except Exception as e:
        lines.append(f"FATAL ERROR: {e}")
    finally:
        try:
            with open("feed_check_output.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
        except Exception:
            pass
        try:
            mt5.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
