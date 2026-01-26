import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

import config

try:
    import MetaTrader5 as mt5
except Exception:
    mt5 = None

try:
    import plotly.graph_objects as go
except Exception:
    go = None

try:
    from fundamentals import FundamentalFetcher
except Exception:
    FundamentalFetcher = None

try:
    import otimizador_semanal as opt
except Exception:
    opt = None


@dataclass
class RankingConfig:
    d1_window_days: int = 126
    liquidity_days: int = 20
    min_d1_bars: int = 120
    optimizer_output_dir: str = getattr(config, "OPTIMIZER_OUTPUT", "optimizer_output")
    max_assets_report: int = 60
    portfolio_sizes: tuple[int, ...] = (5, 10, 15, 20)
    portfolio_m15_bars: int = 1200


DEFAULT_PARAMS: dict[str, Any] = {
    "ema_short": 9,
    "ema_long": 21,
    "rsi_low": 30,
    "rsi_high": 70,
    "adx_threshold": 25,
    "mom_min": 0.0,
    "sl_atr_multiplier": 2.5,
    "tp_mult": 5.0,
    "base_slippage": 0.0035,
}


def _ensure_output_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if not math.isfinite(v):
            return default
        return v
    except Exception:
        return default


def _try_init_mt5(max_retries: int = 3, wait_seconds: float = 3.0) -> bool:
    if mt5 is None:
        return False

    path = getattr(config, "MT5_TERMINAL_PATH", None)
    for _ in range(max_retries):
        try:
            ok = mt5.initialize(path=path) if path else mt5.initialize()
            if ok and (mt5.terminal_info() and mt5.terminal_info().connected):
                return True
        except Exception:
            pass
        try:
            mt5.shutdown()
        except Exception:
            pass
        time.sleep(wait_seconds)
    return False


def _get_d1_rates(symbol: str, bars: int) -> Optional[pd.DataFrame]:
    if mt5 is None:
        return None
    try:
        mt5.symbol_select(symbol, True)
    except Exception:
        pass

    try:
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_D1, 1, int(bars))
    except Exception:
        rates = None
    if rates is None or len(rates) == 0:
        return None

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.set_index("time").sort_index()
    return df


def _compute_liquidity_financial(df_d1: pd.DataFrame, days: int) -> float:
    if df_d1 is None or df_d1.empty:
        return 0.0

    df = df_d1.tail(int(days)).copy()
    vol_col = None
    if "real_volume" in df.columns and float(df["real_volume"].sum() or 0) > 0:
        vol_col = "real_volume"
    elif "tick_volume" in df.columns:
        vol_col = "tick_volume"
    elif "volume" in df.columns:
        vol_col = "volume"
    if vol_col is None:
        return 0.0

    avg_vol = _safe_float(df[vol_col].mean(), 0.0)
    avg_close = _safe_float(df["close"].mean(), 0.0)
    return max(0.0, avg_vol * avg_close)


def _compute_volatility(df_d1: pd.DataFrame, window: int) -> float:
    if df_d1 is None or df_d1.empty or len(df_d1) < window + 2:
        return 0.0
    close = df_d1["close"].astype(float).dropna()
    rets = close.pct_change().dropna().tail(int(window))
    if len(rets) < 20:
        return 0.0
    vol_daily = float(rets.std())
    return max(0.0, vol_daily * math.sqrt(252))


def _compute_corr_with_ibov(df_sym: pd.DataFrame, df_ibov: pd.DataFrame, window: int) -> float:
    if df_sym is None or df_ibov is None:
        return 0.0
    if df_sym.empty or df_ibov.empty:
        return 0.0

    sym_ret = df_sym["close"].astype(float).pct_change()
    ibov_ret = df_ibov["close"].astype(float).pct_change()
    aligned = pd.concat([sym_ret, ibov_ret], axis=1).dropna().tail(int(window))
    if len(aligned) < 20:
        return 0.0
    c = float(aligned.corr().iloc[0, 1])
    if not math.isfinite(c):
        return 0.0
    return max(-1.0, min(1.0, c))


def _load_elite_params_latest(output_dir: str) -> dict[str, dict[str, Any]]:
    p = Path(output_dir) / "elite_symbols_latest.json"
    if not p.exists():
        return {}
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
        elite = payload.get("elite_symbols") or {}
        out: dict[str, dict[str, Any]] = {}
        for sym, cfg in elite.items():
            if isinstance(cfg, dict):
                out[str(sym).strip().upper()] = cfg
        return out
    except Exception:
        return {}


def build_asset_ranking(rcfg: RankingConfig) -> pd.DataFrame:
    if not _try_init_mt5():
        raise RuntimeError("MT5 indisponível para ranking")

    symbols = sorted({str(s).strip().upper() for s in getattr(config, "SECTOR_MAP", {}).keys() if str(s).strip()})

    df_ibov = _get_d1_rates("IBOV", bars=max(rcfg.d1_window_days + 30, 200))
    if df_ibov is None:
        df_ibov = pd.DataFrame()

    fetcher = FundamentalFetcher() if FundamentalFetcher else None

    rows: list[dict[str, Any]] = []
    for sym in symbols:
        df_d1 = _get_d1_rates(sym, bars=max(rcfg.d1_window_days + 50, 250))
        if df_d1 is None or len(df_d1) < rcfg.min_d1_bars:
            continue

        avg_fin = _compute_liquidity_financial(df_d1, days=rcfg.liquidity_days)
        vol = _compute_volatility(df_d1, window=rcfg.d1_window_days)
        corr = _compute_corr_with_ibov(df_d1, df_ibov, window=rcfg.d1_window_days)

        market_cap = 0.0
        if fetcher is not None:
            try:
                fund = fetcher.get_fundamentals(sym) or {}
                market_cap = _safe_float(fund.get("market_cap", 0.0), 0.0)
            except Exception:
                market_cap = 0.0

        rows.append(
            {
                "symbol": sym,
                "subsetor": (getattr(config, "SUBSETOR_MAP", {}) or {}).get(sym, ""),
                "avg_fin_volume": float(avg_fin),
                "volatility_ann": float(vol),
                "corr_ibov": float(corr),
                "abs_corr_ibov": float(abs(corr)),
                "market_cap": float(market_cap),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    def _pct_rank(series: pd.Series, asc: bool) -> pd.Series:
        s = series.fillna(0.0).astype(float)
        return s.rank(pct=True, ascending=asc)

    df["score_liquidity"] = _pct_rank(df["avg_fin_volume"], asc=False)
    df["score_volatility"] = _pct_rank(df["volatility_ann"], asc=False)
    df["score_market_cap"] = _pct_rank(df["market_cap"], asc=False)
    df["score_corr_diversification"] = _pct_rank(1.0 - df["abs_corr_ibov"], asc=False)

    df["score_total"] = (
        0.35 * df["score_liquidity"]
        + 0.20 * df["score_volatility"]
        + 0.25 * df["score_market_cap"]
        + 0.20 * df["score_corr_diversification"]
    )

    df["rank_total"] = df["score_total"].rank(ascending=False, method="min").astype(int)

    q80 = df["score_total"].quantile(0.80)
    q55 = df["score_total"].quantile(0.55)
    q30 = df["score_total"].quantile(0.30)
    df["tier"] = np.where(
        df["score_total"] >= q80,
        "A",
        np.where(df["score_total"] >= q55, "B", np.where(df["score_total"] >= q30, "C", "D")),
    )

    df = df.sort_values(["rank_total", "symbol"]).reset_index(drop=True)
    return df


def _portfolio_from_equity_curves(equity_curves: list[np.ndarray], initial: float = 100000.0) -> np.ndarray:
    if not equity_curves:
        return np.asarray([initial], dtype=float)

    min_len = min(len(x) for x in equity_curves if x is not None and len(x) > 0)
    if min_len <= 1:
        return np.asarray([initial], dtype=float)

    norm = []
    for eq in equity_curves:
        eq = np.asarray(eq, dtype=float)[:min_len]
        if eq[0] <= 0:
            continue
        norm.append(eq / eq[0])
    if not norm:
        return np.asarray([initial], dtype=float)

    stack = np.vstack(norm)
    mean_curve = stack.mean(axis=0)
    return mean_curve * initial


def run_portfolio_size_study(rcfg: RankingConfig, ranking_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    if opt is None:
        raise RuntimeError("Não foi possível importar otimizador_semanal (backtest indisponível)")
    if not _try_init_mt5():
        raise RuntimeError("MT5 indisponível para backtest de carteiras")

    elite_params = _load_elite_params_latest(rcfg.optimizer_output_dir)
    sizes = [int(x) for x in rcfg.portfolio_sizes]

    results: list[dict[str, Any]] = []
    curves: dict[int, np.ndarray] = {}

    for n in sizes:
        selected = ranking_df.head(n)["symbol"].tolist()
        per_asset_metrics = []
        for sym in selected:
            params = dict(DEFAULT_PARAMS)
            ep = elite_params.get(sym)
            if isinstance(ep, dict):
                params.update({k: ep.get(k) for k in DEFAULT_PARAMS.keys() if k in ep})

            df_m15 = opt.load_data_with_retry(sym, bars=int(rcfg.portfolio_m15_bars), timeframe=mt5.TIMEFRAME_M15) if hasattr(opt, "load_data_with_retry") else None
            if df_m15 is None or len(df_m15) < 200:
                continue

            m = opt.backtest_params_on_df(sym, params, df_m15)
            per_asset_metrics.append(
                {
                    "symbol": sym,
                    "equity_curve": np.asarray(m.get("equity_curve", [100000.0]), dtype=float),
                    "trades": int(m.get("total_trades", 0) or 0),
                    "costs_paid": _safe_float(m.get("costs_paid", 0.0), 0.0),
                }
            )

        eqs = [x["equity_curve"] for x in per_asset_metrics]
        port_eq = _portfolio_from_equity_curves(eqs, initial=100000.0)
        curves[n] = port_eq

        m_port = opt.compute_advanced_metrics(port_eq.tolist()) if hasattr(opt, "compute_advanced_metrics") else {}
        total_trades = sum(x["trades"] for x in per_asset_metrics)
        costs_paid = sum(x["costs_paid"] for x in per_asset_metrics)

        results.append(
            {
                "n_assets": n,
                "n_assets_effective": len(per_asset_metrics),
                "total_return": _safe_float(m_port.get("total_return", 0.0), 0.0),
                "sharpe": _safe_float(m_port.get("sharpe", 0.0), 0.0),
                "sortino": _safe_float(m_port.get("sortino", 0.0), 0.0),
                "calmar": _safe_float(m_port.get("calmar", 0.0), 0.0),
                "max_drawdown": _safe_float(m_port.get("max_drawdown", 1.0), 1.0),
                "profit_factor": _safe_float(m_port.get("profit_factor", 0.0), 0.0),
                "total_trades": int(total_trades),
                "costs_paid_proxy": float(costs_paid),
                "final_equity": float(port_eq[-1]) if len(port_eq) else 100000.0,
            }
        )

    df_res = pd.DataFrame(results).sort_values("n_assets").reset_index(drop=True)
    return df_res, curves


def _save_plotly_html(fig, out_path: str):
    if fig is None:
        return
    try:
        fig.write_html(out_path, include_plotlyjs="cdn")
    except Exception:
        pass


def generate_reports(rcfg: RankingConfig, ranking_df: pd.DataFrame, portfolio_df: pd.DataFrame, portfolio_curves: dict[int, np.ndarray]) -> dict[str, str]:
    out_dir = _ensure_output_dir(rcfg.optimizer_output_dir)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    ranking_csv = os.path.join(out_dir, f"b3_asset_ranking_{ts}.csv")
    ranking_latest = os.path.join(out_dir, "b3_asset_ranking_latest.csv")
    ranking_df.to_csv(ranking_csv, index=False)
    ranking_df.to_csv(ranking_latest, index=False)

    ranking_json = os.path.join(out_dir, f"b3_asset_ranking_{ts}.json")
    ranking_latest_json = os.path.join(out_dir, "b3_asset_ranking_latest.json")
    try:
        cols = [
            "symbol",
            "subsetor",
            "tier",
            "rank_total",
            "score_total",
            "avg_fin_volume",
            "volatility_ann",
            "corr_ibov",
            "market_cap",
        ]
        compact = ranking_df[cols].copy() if all(c in ranking_df.columns for c in cols) else ranking_df.copy()
        tiers = {}
        for t in ["A", "B", "C", "D"]:
            tdf = compact[compact.get("tier") == t] if "tier" in compact.columns else compact
            tiers[t] = tdf.sort_values(["rank_total", "symbol"]).to_dict(orient="records") if not tdf.empty else []
        payload = {"generated_at": datetime.now().isoformat(), "tiers": tiers}
        Path(ranking_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        Path(ranking_latest_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

    portfolio_csv = os.path.join(out_dir, f"portfolio_size_study_{ts}.csv")
    portfolio_latest = os.path.join(out_dir, "portfolio_size_study_latest.csv")
    if portfolio_df is not None and not portfolio_df.empty:
        portfolio_df.to_csv(portfolio_csv, index=False)
        portfolio_df.to_csv(portfolio_latest, index=False)

    artifacts: dict[str, str] = {
        "ranking_csv": ranking_csv,
        "ranking_latest": ranking_latest,
        "ranking_json": ranking_json,
        "ranking_latest_json": ranking_latest_json,
        "portfolio_csv": portfolio_csv,
        "portfolio_latest": portfolio_latest,
    }

    if go is not None and ranking_df is not None and not ranking_df.empty:
        top = ranking_df.head(min(rcfg.max_assets_report, len(ranking_df))).copy()
        fig_rank = go.Figure(
            data=[
                go.Scatter(
                    x=top["avg_fin_volume"],
                    y=top["volatility_ann"],
                    mode="markers",
                    marker={
                        "size": np.clip(np.sqrt(np.maximum(top["market_cap"].fillna(0).values, 0.0)) / 1e6, 8, 28),
                        "color": top["tier"].map({"A": 0, "B": 1, "C": 2, "D": 3}).fillna(3),
                        "colorscale": "Viridis",
                        "showscale": False,
                        "opacity": 0.85,
                    },
                    text=top["symbol"],
                    customdata=np.stack([top["corr_ibov"], top["market_cap"]], axis=1),
                    hovertemplate=(
                        "<b>%{text}</b><br>"
                        "Liquidez (R$): %{x:,.0f}<br>"
                        "Vol (ann): %{y:.2%}<br>"
                        "Corr IBOV: %{customdata[0]:.2f}<br>"
                        "Market Cap: %{customdata[1]:,.0f}<extra></extra>"
                    ),
                )
            ]
        )
        fig_rank.update_layout(
            title="Ranking B3: Liquidez vs Volatilidade (cor=tier, tamanho=market cap)",
            xaxis_title="Liquidez média (R$ por dia, proxy 20D)",
            yaxis_title="Volatilidade anualizada (proxy D1)",
            template="plotly_white",
            height=600,
        )
        rank_html = os.path.join(out_dir, f"b3_asset_ranking_chart_{ts}.html")
        rank_latest_html = os.path.join(out_dir, "b3_asset_ranking_chart_latest.html")
        _save_plotly_html(fig_rank, rank_html)
        _save_plotly_html(fig_rank, rank_latest_html)
        artifacts["ranking_chart_html"] = rank_html
        artifacts["ranking_chart_latest"] = rank_latest_html

    if go is not None and portfolio_df is not None and not portfolio_df.empty and portfolio_curves:
        fig_eq = go.Figure()
        for n, eq in sorted(portfolio_curves.items(), key=lambda kv: kv[0]):
            fig_eq.add_trace(go.Scatter(y=eq, mode="lines", name=f"{n} ativos"))
        fig_eq.update_layout(
            title="Backtest comparativo por tamanho de carteira (equity curve agregada)",
            xaxis_title="Barra (M15)",
            yaxis_title="Equity (R$)",
            template="plotly_white",
            height=600,
        )
        eq_html = os.path.join(out_dir, f"portfolio_size_equity_{ts}.html")
        eq_latest = os.path.join(out_dir, "portfolio_size_equity_latest.html")
        _save_plotly_html(fig_eq, eq_html)
        _save_plotly_html(fig_eq, eq_latest)
        artifacts["portfolio_equity_html"] = eq_html
        artifacts["portfolio_equity_latest"] = eq_latest

        fig_metrics = go.Figure()
        fig_metrics.add_trace(go.Bar(x=portfolio_df["n_assets"], y=portfolio_df["sharpe"], name="Sharpe"))
        fig_metrics.add_trace(go.Bar(x=portfolio_df["n_assets"], y=portfolio_df["max_drawdown"], name="Max DD"))
        fig_metrics.update_layout(
            title="Sharpe e Max Drawdown por tamanho de carteira",
            barmode="group",
            xaxis_title="N ativos",
            template="plotly_white",
            height=520,
        )
        met_html = os.path.join(out_dir, f"portfolio_size_metrics_{ts}.html")
        met_latest = os.path.join(out_dir, "portfolio_size_metrics_latest.html")
        _save_plotly_html(fig_metrics, met_html)
        _save_plotly_html(fig_metrics, met_latest)
        artifacts["portfolio_metrics_html"] = met_html
        artifacts["portfolio_metrics_latest"] = met_latest

    return artifacts


def recommend_portfolio_size(portfolio_df: pd.DataFrame) -> dict[str, Any]:
    if portfolio_df is None or portfolio_df.empty:
        return {"recommended_n": None, "reason": "Sem dados"}

    df = portfolio_df.copy()
    df["score"] = (
        df["sharpe"].fillna(0.0)
        - 1.25 * df["max_drawdown"].fillna(1.0)
        + 0.15 * df["calmar"].fillna(0.0)
    )
    best = df.sort_values(["score", "sharpe"], ascending=False).iloc[0]
    return {
        "recommended_n": int(best["n_assets"]),
        "n_assets_effective": int(best.get("n_assets_effective", 0) or 0),
        "sharpe": float(best.get("sharpe", 0.0) or 0.0),
        "max_drawdown": float(best.get("max_drawdown", 1.0) or 1.0),
        "calmar": float(best.get("calmar", 0.0) or 0.0),
        "reason": "Maximiza score (Sharpe - 1.25*DD + 0.15*Calmar) com custos embutidos no backtest",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["rank", "portfolio", "full"], default="full")
    parser.add_argument("--schedule", action="store_true")
    parser.add_argument("--d1-window-days", type=int, default=126)
    parser.add_argument("--portfolio-bars", type=int, default=1200)
    args = parser.parse_args()

    rcfg = RankingConfig(d1_window_days=int(args.d1_window_days), portfolio_m15_bars=int(args.portfolio_bars))

    def _run_once():
        ranking_df = build_asset_ranking(rcfg)
        portfolio_df = pd.DataFrame()
        curves: dict[int, np.ndarray] = {}
        if args.mode in ("portfolio", "full"):
            portfolio_df, curves = run_portfolio_size_study(rcfg, ranking_df)
        artifacts = generate_reports(rcfg, ranking_df, portfolio_df, curves)
        rec = recommend_portfolio_size(portfolio_df) if args.mode in ("portfolio", "full") else {}
        out_dir = _ensure_output_dir(rcfg.optimizer_output_dir)
        (Path(out_dir) / "portfolio_size_recommendation_latest.json").write_text(
            json.dumps({"generated_at": datetime.now().isoformat(), "recommendation": rec}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return artifacts, rec

    if args.schedule:
        try:
            import schedule
        except Exception:
            _run_once()
            return

        schedule.every().sunday.at("22:00").do(_run_once)
        while True:
            schedule.run_pending()
            time.sleep(30)
    else:
        _run_once()


if __name__ == "__main__":
    main()
