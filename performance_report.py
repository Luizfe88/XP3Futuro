import argparse
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

import database


@dataclass
class CostModel:
    fixed_brl: float = 2.0
    pct_notional: float = 0.0003

    def estimate_cost(self, notional_brl: float) -> float:
        return max(self.fixed_brl, abs(notional_brl) * self.pct_notional)


def _session_from_hour(hour: int) -> str:
    if hour < 12:
        return "Manhã"
    if 12 <= hour < 14:
        return "Almoço"
    return "Tarde"


def _prepare(df: pd.DataFrame, cost: CostModel) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    out = out[out["exit_price"].notna()].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out = out[out["timestamp"].notna()].copy()

    out["hour"] = out["timestamp"].dt.hour
    out["session"] = out["hour"].apply(_session_from_hour)
    out["notional"] = out["volume"].astype(float) * out["entry_price"].astype(float)
    out["est_cost"] = out["notional"].apply(cost.estimate_cost)
    out["pnl_net"] = out["pnl_money"].astype(float) - out["est_cost"]
    out["win_net"] = (out["pnl_net"] > 0).astype(int)

    return out


def _summary_block(df: pd.DataFrame, title: str) -> str:
    if df.empty:
        return f"{title}\n- sem trades\n"

    total = int(len(df))
    wins = int(df["win_net"].sum())
    win_rate = (wins / total) * 100 if total else 0.0
    pnl = float(df["pnl_net"].sum())
    avg = float(df["pnl_net"].mean())

    gross_profit = float(df.loc[df["pnl_net"] > 0, "pnl_net"].sum())
    gross_loss = float(-df.loc[df["pnl_net"] < 0, "pnl_net"].sum())
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    return (
        f"{title}\n"
        f"- trades: {total} | wins: {wins} | win_rate líquido: {win_rate:.1f}%\n"
        f"- pnl líquido: R$ {pnl:,.2f} | médio/trade: R$ {avg:,.2f} | profit_factor: {profit_factor:.2f}\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--fixed-cost", type=float, default=2.0)
    parser.add_argument("--pct-cost", type=float, default=0.0003)
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()

    cost = CostModel(fixed_brl=args.fixed_cost, pct_notional=args.pct_cost)

    report = database.get_win_rate_report(lookback_days=args.days)
    cutoff = (datetime.now().date()).strftime("%Y-%m-%d")

    import sqlite3

    conn = sqlite3.connect(database.DB_PATH)
    df = pd.read_sql_query(
        "SELECT * FROM trades WHERE date(timestamp) >= date(?) AND exit_price IS NOT NULL",
        conn,
        params=((datetime.now() - pd.Timedelta(days=args.days)).strftime("%Y-%m-%d"),),
    )
    conn.close()

    d = _prepare(df, cost)

    print("=" * 80)
    print(f"XP3 — Performance líquida (estimada) | lookback={args.days}d")
    print(f"Custo: max(R$ {cost.fixed_brl:.2f}, {cost.pct_notional*100:.3f}% do financeiro)")
    print("=" * 80)

    print(_summary_block(d, "Geral"))

    if d.empty:
        print("=" * 80)
        print(f"DB report (pnl bruto, sem custo): total_trades={report.get('total_trades', 0)} | win_rate={report.get('win_rate', 0):.1f}%")
        return

    by_session = d.groupby("session", dropna=False)
    for sess, g in by_session:
        print(_summary_block(g, f"Por sessão — {sess}"))

    if not d.empty:
        by_symbol = (
            d.groupby("symbol")
            .agg(trades=("symbol", "size"), pnl_net=("pnl_net", "sum"), win_rate=("win_net", "mean"))
            .sort_values(["pnl_net", "trades"], ascending=[False, False])
        )
        print("Top símbolos (pnl líquido):")
        for sym, row in by_symbol.head(args.top).iterrows():
            print(f"- {sym}: pnl R$ {row['pnl_net']:,.2f} | trades {int(row['trades'])} | WR {row['win_rate']*100:.1f}%")

        print("Piores símbolos (pnl líquido):")
        for sym, row in by_symbol.tail(args.top).iterrows():
            print(f"- {sym}: pnl R$ {row['pnl_net']:,.2f} | trades {int(row['trades'])} | WR {row['win_rate']*100:.1f}%")

        by_strategy = (
            d.groupby("strategy")
            .agg(trades=("strategy", "size"), pnl_net=("pnl_net", "sum"), win_rate=("win_net", "mean"))
            .sort_values(["pnl_net", "trades"], ascending=[False, False])
        )
        print("Por estratégia:")
        for strat, row in by_strategy.iterrows():
            print(f"- {strat}: pnl R$ {row['pnl_net']:,.2f} | trades {int(row['trades'])} | WR {row['win_rate']*100:.1f}%")

        if "ab_group" in d.columns:
            by_ab = (
                d.groupby("ab_group")
                .agg(trades=("ab_group", "size"), pnl_net=("pnl_net", "sum"), win_rate=("win_net", "mean"))
                .sort_values(["pnl_net", "trades"], ascending=[False, False])
            )
            print("Por grupo A/B:")
            for grp, row in by_ab.iterrows():
                print(f"- {grp}: pnl R$ {row['pnl_net']:,.2f} | trades {int(row['trades'])} | WR {row['win_rate']*100:.1f}%")

    print("=" * 80)
    print(f"DB report (pnl bruto, sem custo): total_trades={report.get('total_trades', 0)} | win_rate={report.get('win_rate', 0):.1f}%")


if __name__ == "__main__":
    main()
