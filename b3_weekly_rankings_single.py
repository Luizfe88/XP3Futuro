"""
B3 Weekly Rankings - Single File Executable

Objetivo:
  - Classificar ativos da B3 por:
      1) Liquidez diária (volume financeiro)
      2) Volatilidade histórica (desvio padrão de retornos)
      3) Correlação com IBOVESPA
      4) Capitalização de mercado (market cap)
  - Rodar estudo de tamanho de carteira (5/10/15/20) com:
      - Sharpe
      - Máximo drawdown
      - Proxy de custos operacionais (taxas/emolumentos)
  - Gerar relatórios e gráficos em optimizer_output/

Dependências (pip):
  - MetaTrader5
  - numpy
  - pandas
  - plotly
  - yfinance
  - schedule (opcional, apenas para agendamento)
"""

from __future__ import annotations

import argparse
import contextlib
import io
import importlib.util
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

try:
    import MetaTrader5 as mt5
except Exception as e:
    raise SystemExit(f"Dependência ausente/erro ao importar MetaTrader5: {e}")

try:
    import plotly.graph_objects as go
except Exception as e:
    raise SystemExit(f"Dependência ausente/erro ao importar plotly: {e}")

try:
    import yfinance as yf
except Exception as e:
    raise SystemExit(f"Dependência ausente/erro ao importar yfinance: {e}")


# ======================================================================================
# CONFIGURAÇÃO (B3) - embutida para tornar o arquivo autocontido
# ======================================================================================

OPTIMIZER_OUTPUT = "optimizer_output"
MARKET_CAP_CACHE_FILE = "fundamentals_cache.json"
YAHOO_CACHE_DIRNAME = "yahoo_cache"

SUBSETOR_MAP = {
    "ITUB4": "BANCOS",
    "BBDC4": "BANCOS",
    "BBDC3": "BANCOS",
    "BBAS3": "BANCOS",
    "SANB11": "BANCOS",
    "BPAN4": "BANCOS",
    "B3SA3": "CORRETORAS",
    "BPAC11": "CORRETORAS",
    "IRBR3": "SEGUROS",
    "PSSA3": "SEGUROS",
}

SECTOR_MAP = {
    "ITUB4": "FINANCEIRO",
    "BBDC4": "FINANCEIRO",
    "BBAS3": "FINANCEIRO",
    "B3SA3": "FINANCEIRO",
    "BPAC11": "FINANCEIRO",
    "ITSA4": "FINANCEIRO",
    "ABCB4": "FINANCEIRO",
    "PINE4": "FINANCEIRO",
    "PETR4": "ENERGIA",
    "PRIO3": "ENERGIA",
    "RECV3": "ENERGIA",
    "VBBR3": "ENERGIA",
    "ELET3": "ENERGIA",
    "EQTL3": "ENERGIA",
    "ENEV3": "ENERGIA",
    "NEOE3": "ENERGIA",
    "VALE3": "MATERIAIS BÁSICOS",
    "GGBR4": "MATERIAIS BÁSICOS",
    "USIM5": "MATERIAIS BÁSICOS",
    "CSNA3": "MATERIAIS BÁSICOS",
    "SUZB3": "MATERIAIS BÁSICOS",
    "KLBN11": "MATERIAIS BÁSICOS",
    "AURA33": "MATERIAIS BÁSICOS",
    "ABEV3": "CONSUMO NÃO CÍCLICO",
    "JBSS3": "CONSUMO NÃO CÍCLICO",
    "BRFS3": "CONSUMO NÃO CÍCLICO",
    "BEEF3": "CONSUMO NÃO CÍCLICO",
    "CRFB3": "CONSUMO NÃO CÍCLICO",
    "SLCE3": "CONSUMO NÃO CÍCLICO",
    "RAIZ4": "CONSUMO NÃO CÍCLICO",
    "RDOR3": "SAÚDE",
    "HAPV3": "SAÚDE",
    "RADL3": "SAÚDE",
    "ONCO3": "SAÚDE",
    "QUAL3": "SAÚDE",
    "ANIM3": "SAÚDE",
    "LREN3": "CONSUMO CÍCLICO",
    "MGLU3": "CONSUMO CÍCLICO",
    "YDUQ3": "CONSUMO CÍCLICO",
    "COGN3": "CONSUMO CÍCLICO",
    "CYRE3": "CONSUMO CÍCLICO",
    "MRVE3": "CONSUMO CÍCLICO",
    "TEND3": "CONSUMO CÍCLICO",
    "MDNE3": "CONSUMO CÍCLICO",
    "WEGE3": "INDUSTRIAL",
    "RENT3": "INDUSTRIAL",
    "MOVI3": "INDUSTRIAL",
    "RAIL3": "INDUSTRIAL",
    "CCRO3": "INDUSTRIAL",
    "AZUL4": "INDUSTRIAL",
    "TOTS3": "TECNOLOGIA",
    "LWSA3": "TECNOLOGIA",
    "DESK3": "TECNOLOGIA",
    "VIVT3": "COMUNICAÇÕES",
    "TIMS3": "COMUNICAÇÕES",
    "MULT3": "IMOBILIÁRIO",
    "IGTI11": "IMOBILIÁRIO",
    "SBSP3": "UTILIDADES",
    "BBSE3": "SEGUROS",
    "ODPV3": "SAÚDE",
}

BLUE_CHIPS = [
    "VALE3",
    "PETR4",
    "PETR3",
    "ITUB4",
    "BBDC4",
    "BBAS3",
    "ABEV3",
    "WEGE3",
    "B3SA3",
    "ELET3",
    "SUZB3",
    "JBSS3",
]


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
    "risk_per_trade": 0.01,
}


@dataclass
class RankingConfig:
    d1_window_days: int = 126
    liquidity_days: int = 20
    min_d1_bars: int = 120
    max_assets_report: int = 60
    portfolio_sizes: tuple[int, ...] = (5, 10, 15, 20)
    portfolio_m15_bars: int = 1200
    output_dir: str = OPTIMIZER_OUTPUT


# ======================================================================================
# UTILITÁRIOS (segurança, normalização, cache)
# ======================================================================================

def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _mt5_initialize_with_retry(max_retries: int = 3, wait_seconds: float = 3.0) -> bool:
    terminal_path = os.getenv("MT5_TERMINAL_PATH", "").strip() or None
    for _ in range(max_retries):
        try:
            ok = mt5.initialize(path=terminal_path) if terminal_path else mt5.initialize()
            term = mt5.terminal_info()
            if ok and term and term.connected:
                return True
        except Exception:
            pass
        try:
            mt5.shutdown()
        except Exception:
            pass
        time.sleep(wait_seconds)
    return False


def _get_rates_mt5(symbol: str, timeframe: int, bars: int, start_pos: int = 0) -> Optional[pd.DataFrame]:
    try:
        mt5.symbol_select(symbol, True)
    except Exception:
        pass

    rates = None
    try:
        rates = mt5.copy_rates_from_pos(symbol, timeframe, int(start_pos), int(bars))
    except Exception:
        rates = None
    if rates is None or len(rates) == 0:
        return None

    df = pd.DataFrame(rates)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df = df.set_index("time")
    df = df.sort_index()

    if "tick_volume" in df.columns and "volume" not in df.columns:
        df = df.rename(columns={"tick_volume": "volume"})
    if "real_volume" in df.columns and "volume" not in df.columns:
        df = df.rename(columns={"real_volume": "volume"})
    if "volume" not in df.columns:
        df["volume"] = 1.0
    df["volume"] = df["volume"].fillna(0)

    cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[cols]
    return df


def _to_yf_ticker(symbol: str) -> str:
    sym = str(symbol).strip().upper()
    if sym in ("IBOV", "IBOVESPA", "^BVSP"):
        return "^BVSP"
    return f"{sym}.SA"


def _timeframe_to_yf_interval(timeframe: int) -> str:
    if timeframe == mt5.TIMEFRAME_D1:
        return "1d"
    if timeframe == mt5.TIMEFRAME_M15:
        return "15m"
    return "1d"


def _yf_download_cached(ticker: str, interval: str, period: str, cache_dir: str) -> Optional[pd.DataFrame]:
    _ensure_dir(cache_dir)
    safe_name = ticker.replace("^", "").replace("/", "_").replace(":", "_")
    cache_file = os.path.join(cache_dir, f"{safe_name}_{interval}_{period}.csv")

    if os.path.exists(cache_file):
        try:
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            if df is not None and not df.empty:
                return df
        except Exception:
            pass

    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            df = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False, threads=False)
    except Exception:
        return None

    if df is None or df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )

    for c in ["open", "high", "low", "close"]:
        if c not in df.columns:
            return None

    if "volume" not in df.columns:
        df["volume"] = 0.0

    df = df[["open", "high", "low", "close", "volume"]].dropna()
    try:
        df.to_csv(cache_file)
    except Exception:
        pass
    return df


def _get_rates_yahoo(symbol: str, timeframe: int, bars: int, cache_dir: str) -> Optional[pd.DataFrame]:
    interval = _timeframe_to_yf_interval(timeframe)
    ticker = _to_yf_ticker(symbol)
    if interval == "15m":
        period = "60d"
    else:
        period = "1y"

    df = _yf_download_cached(ticker, interval=interval, period=period, cache_dir=cache_dir)
    if df is None or df.empty:
        return None

    if bars > 0 and len(df) > bars:
        df = df.tail(int(bars))
    return df


def _get_rates(symbol: str, timeframe: int, bars: int, start_pos: int = 0, source: str = "auto", output_dir: str = OPTIMIZER_OUTPUT) -> Optional[pd.DataFrame]:
    src = str(source).strip().lower()
    cache_dir = os.path.join(output_dir, YAHOO_CACHE_DIRNAME)

    if src == "mt5":
        return _get_rates_mt5(symbol, timeframe, bars, start_pos=start_pos)
    if src in ("yahoo", "yf"):
        return _get_rates_yahoo(symbol, timeframe, bars, cache_dir=cache_dir)

    info = None
    try:
        info = mt5.symbol_info(str(symbol).strip().upper())
    except Exception:
        info = None

    if info is not None:
        df = _get_rates_mt5(symbol, timeframe, bars, start_pos=start_pos)
        if df is not None and not df.empty:
            return df

    return _get_rates_yahoo(symbol, timeframe, bars, cache_dir=cache_dir)


class MarketCapCache:
    def __init__(self, cache_file: str):
        self.cache_file = cache_file
        self._cache: dict[str, Any] = {}
        self._load()

    def _load(self):
        try:
            if os.path.exists(self.cache_file):
                self._cache = json.loads(Path(self.cache_file).read_text(encoding="utf-8"))
        except Exception:
            self._cache = {}

    def _save(self):
        try:
            Path(self.cache_file).write_text(json.dumps(self._cache, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    def get_market_cap(self, symbol: str, ttl_hours: int = 72) -> float:
        sym = str(symbol).strip().upper()
        now_ts = time.time()
        entry = self._cache.get(sym)
        if isinstance(entry, dict):
            ts = _safe_float(entry.get("ts", 0.0), 0.0)
            if ts > 0 and (now_ts - ts) <= ttl_hours * 3600:
                return _safe_float(entry.get("market_cap", 0.0), 0.0)

        yf_symbol = f"{sym}.SA"
        market_cap = 0.0
        try:
            info = yf.Ticker(yf_symbol).info
            market_cap = _safe_float(info.get("marketCap", 0.0), 0.0)
        except Exception:
            market_cap = 0.0

        self._cache[sym] = {"market_cap": market_cap, "ts": now_ts}
        self._save()
        return market_cap


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


# ======================================================================================
# MÉTRICAS (volatilidade, correlação, sharpe, drawdown)
# ======================================================================================

def _compute_liquidity_financial(df_d1: pd.DataFrame, days: int) -> float:
    if df_d1 is None or df_d1.empty:
        return 0.0
    df = df_d1.tail(int(days)).copy()
    avg_vol = _safe_float(df["volume"].mean(), 0.0)
    avg_close = _safe_float(df["close"].mean(), 0.0)
    return max(0.0, avg_vol * avg_close)

def _compute_liquidity_components(df_d1: pd.DataFrame, days: int) -> tuple[float, float, float]:
    if df_d1 is None or df_d1.empty:
        return 0.0, 0.0, 0.0
    df = df_d1.tail(int(days)).copy()
    avg_vol = _safe_float(df["volume"].mean(), 0.0)
    avg_close = _safe_float(df["close"].mean(), 0.0)
    return float(avg_vol), float(avg_close), float(max(0.0, avg_vol * avg_close))


def _compute_volatility(df_d1: pd.DataFrame, window: int) -> float:
    if df_d1 is None or df_d1.empty or len(df_d1) < window + 2:
        return 0.0
    close = df_d1["close"].astype(float).dropna()
    rets = close.pct_change().dropna().tail(int(window))
    if len(rets) < 20:
        return 0.0
    return max(0.0, float(rets.std()) * math.sqrt(252))


def _compute_corr(df_sym: pd.DataFrame, df_ibov: pd.DataFrame, window: int) -> float:
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


def compute_advanced_metrics(equity_curve: list[float], bars_per_year: int = 26 * 252, risk_free_rate: float = 0.0) -> dict[str, Any]:
    if not equity_curve or len(equity_curve) < 2:
        return {"total_return": 0.0, "max_drawdown": 0.01, "calmar": 0.0, "sortino": 0.0, "sharpe": 0.0}

    eq = np.asarray(equity_curve, dtype=float)
    rets = np.diff(eq) / np.maximum(eq[:-1], 1e-12)
    if len(rets) < 50:
        return {"total_return": float(eq[-1] / eq[0] - 1), "max_drawdown": 1.0, "calmar": 0.0, "sortino": 0.0, "sharpe": 0.0}

    total_return = float(eq[-1] / max(eq[0], 1e-12) - 1)
    peak = np.maximum.accumulate(eq)
    drawdowns = (eq - peak) / np.maximum(peak, 1e-12)
    max_dd = float(max(-np.min(drawdowns), 0.01))

    avg_ret = float(np.mean(rets))
    ret_std = float(np.std(rets))
    annualized = avg_ret * float(bars_per_year)
    sharpe = (annualized - risk_free_rate) / (ret_std * math.sqrt(bars_per_year)) if ret_std > 0 else 0.0

    downside = rets[rets < 0]
    downside_std = float(np.std(downside)) * math.sqrt(bars_per_year) if len(downside) > 0 else 0.0
    sortino = (annualized - risk_free_rate) / downside_std if downside_std > 0 else 0.0
    calmar = annualized / max_dd if max_dd > 0 else 0.0

    return {
        "total_return": float(total_return),
        "max_drawdown": float(max_dd),
        "calmar": float(calmar),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "final_equity": float(eq[-1]),
    }


# ======================================================================================
# BACKTEST (M15) - simplificado, autocontido, com proxy de custos B3
# ======================================================================================

def calculate_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> tuple[np.ndarray, np.ndarray]:
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    close = np.asarray(close, dtype=float)

    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum.reduce([tr1, tr2, tr3])

    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    atr = pd.Series(tr).ewm(alpha=1 / period, adjust=False).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1 / period, adjust=False).mean().values / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1 / period, adjust=False).mean().values / (atr + 1e-10)

    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1 / period, adjust=False).mean().fillna(0).values

    return adx, atr


def backtest_params_on_df(params: dict[str, Any], df: pd.DataFrame) -> dict[str, Any]:
    if df is None or df.empty or len(df) < 150:
        return {"equity_curve": [100000.0], "total_trades": 0, "costs_paid": 0.0}

    close = df["close"].values.astype(float)
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)
    volume = df["volume"].values.astype(float)

    ema_s = pd.Series(close).ewm(span=int(params.get("ema_short", 9)), adjust=False).mean().values
    ema_l = pd.Series(close).ewm(span=int(params.get("ema_long", 21)), adjust=False).mean().values

    delta = pd.Series(close).diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)
    rsi = (100 - (100 / (1 + rs))).fillna(50).values

    delta_2 = pd.Series(close).diff()
    gain_2 = (delta_2.where(delta_2 > 0, 0)).rolling(window=2).mean()
    loss_2 = (-delta_2.where(delta_2 < 0, 0)).rolling(window=2).mean()
    rs_2 = gain_2 / (loss_2 + 1e-10)
    rsi_2 = (100 - (100 / (1 + rs_2))).fillna(50).values

    momentum = pd.Series(close).pct_change(periods=10).fillna(0).values
    adx, atr = calculate_adx(high, low, close, period=14)

    cash = 100000.0
    equity = cash
    position_notional = 0.0
    entry_price = 0.0
    stop_price = 0.0
    target_price = 0.0
    state = 0

    trades = 0
    wins = 0
    losses = 0
    costs_paid = 0.0

    transaction_cost_pct = 0.0003 + 0.00025
    base_slippage = float(params.get("base_slippage", 0.0035) or 0.0035)
    rsi_low = float(params.get("rsi_low", 30) or 30)
    adx_threshold = float(params.get("adx_threshold", 25) or 25)
    sl_mult = float(params.get("sl_atr_multiplier", 2.5) or 2.5)
    tp_mult = float(params.get("tp_mult", 5.0) or 5.0)
    risk_per_trade = float(params.get("risk_per_trade", 0.01) or 0.01)

    equity_curve = np.full(len(close), cash, dtype=float)

    for i in range(1, len(close)):
        current_price = float(close[i])

        if state == 1:
            if float(high[i]) >= target_price:
                exit_price = target_price
                gross_profit = (exit_price - entry_price) * (position_notional / max(entry_price, 1e-12))
                cost = (position_notional * transaction_cost_pct) + ((position_notional + gross_profit) * transaction_cost_pct)
                net_profit = gross_profit - cost
                cash += position_notional + net_profit
                costs_paid += cost
                if net_profit > 0:
                    wins += 1
                else:
                    losses += 1
                trades += 1
                state = 0
                position_notional = 0.0
                equity = cash
            elif float(low[i]) <= stop_price:
                exit_price = stop_price - (stop_price * base_slippage)
                gross_profit = (exit_price - entry_price) * (position_notional / max(entry_price, 1e-12))
                cost = (position_notional * transaction_cost_pct) + ((position_notional + gross_profit) * transaction_cost_pct)
                net_profit = gross_profit - cost
                cash += position_notional + net_profit
                costs_paid += cost
                losses += 1
                trades += 1
                state = 0
                position_notional = 0.0
                equity = cash
            else:
                current_val = position_notional * (current_price / max(entry_price, 1e-12))
                equity = cash + current_val
        else:
            trend_condition = current_price > float(ema_l[i])
            pullback_condition = float(rsi[i]) < rsi_low
            reversion_condition = float(rsi_2[i]) < 5.0
            volatility_ok = float(adx[i]) > adx_threshold
            signal = (trend_condition and pullback_condition and volatility_ok) or reversion_condition

            if signal:
                sl_dist = float(atr[i]) * sl_mult
                tp_dist = float(atr[i]) * tp_mult
                entry_price = current_price * (1 + base_slippage)
                stop_price = entry_price - sl_dist
                target_price = entry_price + tp_dist

                risk_amt = equity * risk_per_trade
                if sl_dist > 0:
                    shares_raw = risk_amt / sl_dist
                    cost_basis = shares_raw * entry_price
                    if cost_basis > cash * 0.95:
                        cost_basis = cash * 0.95

                    position_notional = cost_basis
                    cash -= position_notional
                    state = 1

        equity_curve[i] = equity

?    if state == 1 and len(close) >= 2:
        exit_price = float(close[-1]) * (1 - base_slippage)
        gross_profit = (exit_price - entry_price) * (position_notional / max(entry_price, 1e-12))
        cost = (position_notional * transaction_cost_pct) + ((position_notional + gross_profit) * transaction_cost_pct)
        net_profit = gross_profit - cost
        cash += position_notional + net_profit
        costs_paid += cost
        if net_profit > 0:
            wins += 1
        else:
            losses += 1
        trades += 1
        position_notional = 0.0
        equity_curve[-1] = cash

    metrics = compute_advanced_metrics(equity_curve.tolist())
    metrics.update({"equity_curve": equity_curve.tolist(), "total_trades": int(trades), "wins": int(wins), "losses": int(losses), "costs_paid": float(costs_paid)})
    return metrics


# ======================================================================================
# RANKING + ESTUDO DE CARTEIRA + RELATÓRIOS
# ======================================================================================

def build_asset_ranking(rcfg: RankingConfig) -> pd.DataFrame:
    if not _mt5_initialize_with_retry():
        raise RuntimeError("Falha ao conectar ao MT5")

    df_ibov = _get_rates("IBOV", timeframe=mt5.TIMEFRAME_D1, bars=max(rcfg.d1_window_days + 30, 200), start_pos=1, source="auto", output_dir=rcfg.output_dir)
    if df_ibov is None:
        df_ibov = _get_rates("^BVSP", timeframe=mt5.TIMEFRAME_D1, bars=max(rcfg.d1_window_days + 30, 200), start_pos=1, source="yahoo", output_dir=rcfg.output_dir) or pd.DataFrame()
    cap_cache = MarketCapCache(MARKET_CAP_CACHE_FILE)

    rows: list[dict[str, Any]] = []
    missing_mt5 = 0
    used_yahoo = 0
    for sym in sorted(SECTOR_MAP.keys()):
        info = None
        try:
            info = mt5.symbol_info(sym)
        except Exception:
            info = None

        if info is None:
            missing_mt5 += 1
            df_d1 = _get_rates(sym, timeframe=mt5.TIMEFRAME_D1, bars=max(rcfg.d1_window_days + 60, 250), start_pos=1, source="yahoo", output_dir=rcfg.output_dir)
            used_yahoo += 1
            src_used = "yahoo"
        else:
            df_d1 = _get_rates(sym, timeframe=mt5.TIMEFRAME_D1, bars=max(rcfg.d1_window_days + 60, 250), start_pos=1, source="mt5", output_dir=rcfg.output_dir)
            src_used = "mt5"
            if df_d1 is None or df_d1.empty:
                df_d1 = _get_rates(sym, timeframe=mt5.TIMEFRAME_D1, bars=max(rcfg.d1_window_days + 60, 250), start_pos=1, source="yahoo", output_dir=rcfg.output_dir)
                used_yahoo += 1
                src_used = "yahoo"

        if df_d1 is None or df_d1.empty or len(df_d1) < rcfg.min_d1_bars:
            continue

        avg_vol, avg_close, avg_fin = _compute_liquidity_components(df_d1, days=rcfg.liquidity_days)
        last_close = _safe_float(df_d1["close"].iloc[-1], 0.0)
        vol = _compute_volatility(df_d1, window=rcfg.d1_window_days)
        corr = _compute_corr(df_d1, df_ibov, window=rcfg.d1_window_days)
        mcap = cap_cache.get_market_cap(sym)

        rows.append(
            {
                "symbol": sym,
                "data_source": src_used,
                "setor": SECTOR_MAP.get(sym, ""),
                "subsetor": SUBSETOR_MAP.get(sym, ""),
                "avg_volume": float(avg_vol),
                "avg_close": float(avg_close),
                "last_close": float(last_close),
                "d1_bars": int(len(df_d1)),
                "avg_fin_volume": float(avg_fin),
                "volatility_ann": float(vol),
                "corr_ibov": float(corr),
                "abs_corr_ibov": float(abs(corr)),
                "market_cap": float(mcap),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        if missing_mt5 >= max(1, len(SECTOR_MAP) // 2):
            print(
                "⚠️ Nenhum símbolo B3 foi encontrado no MT5 (ex.: PETR4/VALE3/IBOV). "
                "Tentativa via Yahoo Finance também não retornou barras suficientes.",
                flush=True,
            )
        return df

    def pct_rank(series: pd.Series, ascending: bool) -> pd.Series:
        s = series.fillna(0.0).astype(float)
        return s.rank(pct=True, ascending=ascending)

    df["score_liquidity"] = pct_rank(df["avg_fin_volume"], ascending=False)
    df["score_volatility"] = pct_rank(df["volatility_ann"], ascending=False)
    df["score_market_cap"] = pct_rank(df["market_cap"], ascending=False)
    df["score_corr_diversification"] = pct_rank(1.0 - df["abs_corr_ibov"], ascending=False)

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
    df["tier"] = np.where(df["score_total"] >= q80, "A", np.where(df["score_total"] >= q55, "B", np.where(df["score_total"] >= q30, "C", "D")))

    return df.sort_values(["rank_total", "symbol"]).reset_index(drop=True)

def _format_money_br(v: float) -> str:
    try:
        return f"R$ {float(v):,.0f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return "R$ 0"


def _format_number(v: float, decimals: int = 2) -> str:
    try:
        fmt = f"{{:.{int(decimals)}f}}"
        return fmt.format(float(v)).replace(".", ",")
    except Exception:
        return "0"


def _format_pct(v: float, decimals: int = 2) -> str:
    try:
        return f"{float(v) * 100:.{int(decimals)}f}%".replace(".", ",")
    except Exception:
        return "0,00%"


def evaluate_viability(ranking_df: pd.DataFrame, blue_chips: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = ranking_df.copy() if ranking_df is not None else pd.DataFrame()
    ts_iso = datetime.now().isoformat()

    methodology: dict[str, Any] = {
        "generated_at": ts_iso,
        "ranking_criteria": {
            "liquidity": "avg(volume * close) nos últimos 20 pregões (proxy volume financeiro)",
            "volatility": "desvio padrão dos retornos D1 anualizado (janela configurável)",
            "corr_ibov": "correlação dos retornos D1 com IBOV (^BVSP quando necessário)",
            "market_cap": "Yahoo Finance (yfinance) via campo marketCap",
        },
        "score_weights": {"liquidity": 0.35, "volatility": 0.20, "market_cap": 0.25, "diversification": 0.20},
        "viability_rule": {
            "blue_chips_always_viable": True,
            "non_bluechips": "Viável se Liquidez >= mediana e pelo menos 3/4 critérios passam (liq/vol/corr/mcap).",
        },
        "ordering_rule_top10_ex_bluechips": "Ordena por score_total desc; desempate por avg_fin_volume desc.",
        "notes": [
            "Se MT5 não tiver o símbolo B3, usa Yahoo Finance como fallback.",
            "Market cap pode vir como 0 quando indisponível; nesse caso o critério market cap não reprova sozinho.",
        ],
    }

    if df.empty:
        df["is_blue_chip"] = False
        df["viable"] = False
        df["viability_reason"] = "Sem dados suficientes para análise."
        methodology["thresholds"] = {}
        return df, methodology

    df["is_blue_chip"] = df["symbol"].astype(str).str.upper().isin([s.upper() for s in blue_chips])

    liquidity_q50 = float(df["avg_fin_volume"].quantile(0.50))
    vol_q20 = float(df["volatility_ann"].quantile(0.20))
    vol_q90 = float(df["volatility_ann"].quantile(0.90))
    corr_q75 = float(df["abs_corr_ibov"].quantile(0.75))

    mcap_series = df["market_cap"].fillna(0.0).astype(float)
    mcap_pos = mcap_series[mcap_series > 0]
    market_cap_q30 = float(mcap_pos.quantile(0.30)) if len(mcap_pos) >= 10 else 0.0

    methodology["thresholds"] = {
        "liquidity_q50": liquidity_q50,
        "volatility_q20": vol_q20,
        "volatility_q90": vol_q90,
        "abs_corr_q75": corr_q75,
        "market_cap_q30": market_cap_q30,
    }

    viable_list: list[bool] = []
    reasons: list[str] = []
    liquidity_ok_list: list[bool] = []
    volatility_ok_list: list[bool] = []
    corr_ok_list: list[bool] = []
    market_cap_ok_list: list[bool] = []
    passed_list: list[int] = []

    for _, r in df.iterrows():
        sym = str(r.get("symbol", "")).upper()
        is_blue = bool(r.get("is_blue_chip", False))
        avg_fin = float(r.get("avg_fin_volume", 0.0) or 0.0)
        vol = float(r.get("volatility_ann", 0.0) or 0.0)
        abs_corr = float(r.get("abs_corr_ibov", 0.0) or 0.0)
        mcap = float(r.get("market_cap", 0.0) or 0.0)

        if is_blue:
            viable_list.append(True)
            reasons.append("Blue chip: marcada como viável por definição (robustez + liquidez estrutural).")
            liquidity_ok_list.append(True)
            volatility_ok_list.append(True)
            corr_ok_list.append(True)
            market_cap_ok_list.append(True)
            passed_list.append(4)
            continue

        liquidity_ok = avg_fin >= liquidity_q50
        volatility_ok = (vol >= vol_q20) and (vol <= vol_q90)
        corr_ok = abs_corr <= corr_q75
        market_cap_ok = True if market_cap_q30 <= 0 else (mcap >= market_cap_q30)

        passed = int(liquidity_ok) + int(volatility_ok) + int(corr_ok) + int(market_cap_ok)
        viable = bool(liquidity_ok and passed >= 3)

        parts = [
            f"Liquidez {'OK' if liquidity_ok else 'NOK'} ({_format_money_br(avg_fin)}; lim={_format_money_br(liquidity_q50)})",
            f"Vol {'OK' if volatility_ok else 'NOK'} ({_format_pct(vol)}; faixa={_format_pct(vol_q20)}–{_format_pct(vol_q90)})",
            f"Corr {'OK' if corr_ok else 'NOK'} (|corr|={_format_number(abs_corr,3)}; lim={_format_number(corr_q75,3)})",
        ]
        if market_cap_q30 > 0:
            parts.append(f"MCap {'OK' if market_cap_ok else 'NOK'} ({_format_money_br(mcap)}; lim={_format_money_br(market_cap_q30)})")
        else:
            parts.append("MCap N/A (dados insuficientes para threshold confiável)")

        reasons.append("; ".join(parts))
        viable_list.append(viable)
        liquidity_ok_list.append(bool(liquidity_ok))
        volatility_ok_list.append(bool(volatility_ok))
        corr_ok_list.append(bool(corr_ok))
        market_cap_ok_list.append(bool(market_cap_ok))
        passed_list.append(int(passed))

    df["viable"] = viable_list
    df["viability_reason"] = reasons
    df["crit_liquidez_ok"] = liquidity_ok_list
    df["crit_vol_ok"] = volatility_ok_list
    df["crit_corr_ok"] = corr_ok_list
    df["crit_mcap_ok"] = market_cap_ok_list
    df["criteria_passed"] = passed_list
    return df, methodology


def _infer_asset_type(symbol: str) -> str:
    sym = str(symbol).strip().upper()
    if sym.endswith("11"):
        return "UNIT/ETF"
    if sym.endswith("33") or sym.endswith("34") or sym.endswith("35"):
        return "BDR"
    if sym and sym[-1].isdigit():
        return "AÇÃO"
    return "ATIVO"


def _load_elite_symbols_from_config_py(config_py_path: str) -> dict[str, Any]:
    try:
        spec = importlib.util.spec_from_file_location("xp3_config_module", config_py_path)
        if spec is None or spec.loader is None:
            return {}
        module = importlib.util.module_from_spec(spec)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            spec.loader.exec_module(module)
        elite = getattr(module, "ELITE_SYMBOLS", None)
        if isinstance(elite, dict):
            out = {}
            for k, v in elite.items():
                if isinstance(v, dict):
                    out[str(k).strip().upper()] = v
            return out
        return {}
    except Exception:
        return {}


def _build_trade_params_for_symbol(symbol: str, is_blue_chip: bool, elite_symbols: dict[str, Any]) -> dict[str, Any]:
    sym = str(symbol).strip().upper()
    base = dict(DEFAULT_PARAMS)
    base_out: dict[str, Any] = {
        "category": "BLUE CHIP" if is_blue_chip else "OPORTUNIDADE",
        "weight": 0.05,
        "ema_short": int(base.get("ema_short", 9)),
        "ema_long": int(base.get("ema_long", 21)),
        "rsi_low": float(base.get("rsi_low", 30)),
        "rsi_high": float(base.get("rsi_high", 70)),
        "adx_threshold": float(base.get("adx_threshold", 25)),
        "mom_min": float(base.get("mom_min", 0.0)),
        "sl_atr_multiplier": float(base.get("sl_atr_multiplier", 2.5)),
        "tp_mult": float(base.get("tp_mult", 3.0)),
    }

    elite = elite_symbols.get(sym)
    if isinstance(elite, dict):
        base_out["category"] = elite.get("category", base_out["category"])
        if "weight" in elite:
            base_out["weight"] = elite.get("weight")
        for k in ["ema_short", "ema_long", "rsi_low", "rsi_high", "adx_threshold", "mom_min", "sl_atr_multiplier", "tp_mult"]:
            if k in elite:
                base_out[k] = elite.get(k)

    return base_out


def _mt5_backtest_symbol(symbol: str, params: dict[str, Any], bars: int, output_dir: str) -> dict[str, Any]:
    df_m15 = _get_rates(symbol, timeframe=mt5.TIMEFRAME_M15, bars=int(bars), start_pos=0, source="mt5", output_dir=output_dir)
    if df_m15 is None or df_m15.empty or len(df_m15) < 200:
        return {"verified": False, "reason": "Sem barras M15 suficientes no MT5", "bars": int(0)}
    m = backtest_params_on_df(params, df_m15)
    return {
        "verified": True,
        "bars": int(len(df_m15)),
        "total_return": float(m.get("total_return", 0.0) or 0.0),
        "sharpe": float(m.get("sharpe", 0.0) or 0.0),
        "sortino": float(m.get("sortino", 0.0) or 0.0),
        "calmar": float(m.get("calmar", 0.0) or 0.0),
        "max_drawdown": float(m.get("max_drawdown", 1.0) or 1.0),
        "final_equity": float(m.get("final_equity", 100000.0) or 100000.0),
        "total_trades": int(m.get("total_trades", 0) or 0),
        "wins": int(m.get("wins", 0) or 0),
        "losses": int(m.get("losses", 0) or 0),
        "costs_paid_proxy": float(m.get("costs_paid", 0.0) or 0.0),
    }


def build_top20_mt5_verification(rcfg: RankingConfig, ranking_df: pd.DataFrame, mt5_bars: int) -> dict[str, Any]:
    config_py_path = os.path.join(os.path.dirname(__file__), "config.py")
    elite_symbols = _load_elite_symbols_from_config_py(config_py_path)

    if ranking_df is None or ranking_df.empty:
        return {"generated_at": datetime.now().isoformat(), "top20": [], "blue_chips": [], "opportunities": [], "reason": "Ranking vazio"}

    evaluated_df, methodology = evaluate_viability(ranking_df, BLUE_CHIPS)

    def _row_by_symbol(sym: str) -> dict[str, Any]:
        r = evaluated_df[evaluated_df["symbol"] == sym]
        if r.empty:
            return {
                "symbol": sym,
                "setor": SECTOR_MAP.get(sym, ""),
                "tier": "",
                "rank_total": 0,
                "score_total": 0.0,
                "avg_fin_volume": 0.0,
                "volatility_ann": 0.0,
                "abs_corr_ibov": 0.0,
                "market_cap": 0.0,
                "viable": False,
                "data_source": "mt5",
            }
        return r.iloc[0].to_dict()

    blue_candidates = [s for s in BLUE_CHIPS if str(s).upper() in SECTOR_MAP]
    blue_sorted = []
    for sym in blue_candidates:
        sym = str(sym).upper()
        blue_sorted.append(_row_by_symbol(sym))
    if blue_sorted:
        blue_sorted = sorted(blue_sorted, key=lambda r: (float(r.get("market_cap", 0.0) or 0.0), float(r.get("avg_fin_volume", 0.0) or 0.0)), reverse=True)

    blue_top = blue_sorted[:10]
    blue_syms = [str(x.get("symbol", "")).upper() for x in blue_top]
    if len(blue_top) < 10:
        remaining = evaluated_df[~evaluated_df["symbol"].astype(str).str.upper().isin(blue_syms)].copy()
        remaining = remaining.sort_values(["market_cap", "avg_fin_volume"], ascending=[False, False])
        for _, rr in remaining.iterrows():
            if len(blue_top) >= 10:
                break
            sym = str(rr.get("symbol", "")).upper()
            if sym in blue_syms:
                continue
            blue_top.append(rr.to_dict())
            blue_syms.append(sym)

    if not _mt5_initialize_with_retry():
        raise RuntimeError("Falha ao conectar ao MT5")

    rows = []
    for sym in sorted(SECTOR_MAP.keys()):
        row = _row_by_symbol(sym)

        is_blue = str(sym).upper() in blue_syms
        trade_params = _build_trade_params_for_symbol(sym, is_blue, elite_symbols)
        bt = _mt5_backtest_symbol(sym, trade_params, bars=mt5_bars, output_dir=rcfg.output_dir)

        status_ok = False
        if bt.get("verified"):
            sharpe = float(bt.get("sharpe", 0.0) or 0.0)
            dd = float(bt.get("max_drawdown", 1.0) or 1.0)
            trades = int(bt.get("total_trades", 0) or 0)
            status_ok = bool(sharpe >= 0.30 and dd <= 0.35 and trades >= 10)

        rows.append(
            {
                "symbol": sym,
                "setor": row.get("setor", ""),
                "tier": row.get("tier", ""),
                "rank_total": int(row.get("rank_total", 0) or 0),
                "score_total": float(row.get("score_total", 0.0) or 0.0),
                "avg_fin_volume": float(row.get("avg_fin_volume", 0.0) or 0.0),
                "volatility_ann": float(row.get("volatility_ann", 0.0) or 0.0),
                "abs_corr_ibov": float(row.get("abs_corr_ibov", 0.0) or 0.0),
                "market_cap": float(row.get("market_cap", 0.0) or 0.0),
                "viable": bool(row.get("viable", False)),
                "is_blue_chip": bool(is_blue),
                "ativo": trade_params,
                "mt5_verification": bt,
                "mt5_params_ok": bool(status_ok),
            }
        )

    dfv = pd.DataFrame(rows)
    if dfv.empty:
        return {"generated_at": datetime.now().isoformat(), "top20": [], "blue_chips": [], "opportunities": [], "reason": "Sem dados após avaliação"}

    blue_df = dfv[dfv["is_blue_chip"] == True].copy()
    blue_df = blue_df.sort_values(["market_cap", "avg_fin_volume"], ascending=[False, False]).head(10)

    opp_df = dfv[dfv["is_blue_chip"] == False].copy()
    opp_df["verified"] = opp_df["mt5_verification"].apply(lambda x: bool(x.get("verified")) if isinstance(x, dict) else False)
    opp_df["trades"] = opp_df["mt5_verification"].apply(lambda x: int(x.get("total_trades", 0) or 0) if isinstance(x, dict) else 0)
    opp_pref = opp_df[(opp_df["verified"] == True) & (opp_df["trades"] >= 10)].copy()
    if opp_pref.empty:
        opp_pref = opp_df.copy()

    def _sort_key(r: pd.Series):
        bt = r.get("mt5_verification") or {}
        return (
            1 if bt.get("verified") else 0,
            1 if r.get("mt5_params_ok") else 0,
            float(bt.get("sharpe", 0.0) or 0.0),
            float(bt.get("total_return", 0.0) or 0.0),
            -float(bt.get("max_drawdown", 1.0) or 1.0),
            float(r.get("score_total", 0.0) or 0.0),
            float(r.get("avg_fin_volume", 0.0) or 0.0),
        )

    opp_rows = [r for _, r in opp_pref.iterrows()]
    opp_rows = sorted(opp_rows, key=_sort_key, reverse=True)
    opp_top = opp_rows[:10]

    top20 = blue_df.to_dict(orient="records") + [dict(x) for x in opp_top]

    payload = {
        "generated_at": datetime.now().isoformat(),
        "mt5_bars": int(mt5_bars),
        "methodology": {
            "mt5_verification_rule": "Backtest M15 no MT5 com parâmetros do ELITE_SYMBOLS (ou fallback DEFAULT_PARAMS).",
            "mt5_params_ok_thresholds": {"min_sharpe": 0.30, "max_drawdown_max": 0.35, "min_trades": 10},
            "ranking_methodology": methodology,
        },
        "blue_chips": blue_df.to_dict(orient="records"),
        "opportunities": [dict(x) for x in opp_top],
        "top20": top20,
    }
    return payload


def _write_top20_files(out_dir: str, payload: dict[str, Any]) -> dict[str, str]:
    out_dir = _ensure_dir(out_dir)
    ts = payload.get("generated_at") or datetime.now().isoformat()
    json_path = os.path.join(out_dir, "top20_mt5_verification.json")
    txt_path = os.path.join(out_dir, "top20_mt5_verification.txt")

    Path(json_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = []
    lines.append("TOP 20 - MT5 VERIFICAÇÃO (10 BLUE CHIPS + 10 OPORTUNIDADES)")
    lines.append(f"Timestamp: {ts}")
    lines.append(f"Barras M15 (MT5): {payload.get('mt5_bars')}")
    thr = ((payload.get("methodology") or {}).get("mt5_params_ok_thresholds") or {})
    lines.append(f"Critério OK: Sharpe>={thr.get('min_sharpe')} | MaxDD<={thr.get('max_drawdown_max')} | Trades>={thr.get('min_trades')}")
    lines.append("")

    def _fmt_bt(bt: dict[str, Any]) -> str:
        if not bt or not bt.get("verified"):
            return "MT5: N/A"
        return f"MT5 Sharpe={_format_number(bt.get('sharpe',0.0),3)} | DD={_format_pct(bt.get('max_drawdown',1.0),2)} | Ret={_format_pct(bt.get('total_return',0.0),2)} | Trades={int(bt.get('total_trades',0) or 0)}"

    lines.append("BLUE CHIPS (10)")
    for i, r in enumerate(payload.get("blue_chips", [])[:10], start=1):
        bt = r.get("mt5_verification") or {}
        ok = "OK" if r.get("mt5_params_ok") else "NOK"
        lines.append(f"{i:>2}. {r.get('symbol','')} | {ok} | { _fmt_bt(bt) } | liq={_format_money_br(r.get('avg_fin_volume',0.0))}")
    lines.append("")
    lines.append("OPORTUNIDADES (10) - melhores fora blue chips")
    for i, r in enumerate(payload.get("opportunities", [])[:10], start=1):
        bt = r.get("mt5_verification") or {}
        ok = "OK" if r.get("mt5_params_ok") else "NOK"
        lines.append(f"{i:>2}. {r.get('symbol','')} | {ok} | { _fmt_bt(bt) } | score={_format_number(r.get('score_total',0.0),3)} | tier={r.get('tier','')}")

    Path(txt_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"top20_json": json_path, "top20_txt": txt_path}

def build_viable_report_payload(evaluated_df: pd.DataFrame, methodology: dict[str, Any]) -> dict[str, Any]:
    ts_iso = datetime.now().isoformat()
    config_py_path = os.path.join(os.path.dirname(__file__), "config.py")
    elite_symbols = _load_elite_symbols_from_config_py(config_py_path)

    total = int(len(evaluated_df)) if evaluated_df is not None else 0
    viable_df = evaluated_df[evaluated_df["viable"] == True].copy() if evaluated_df is not None and not evaluated_df.empty else pd.DataFrame()
    viable_count = int(len(viable_df)) if not viable_df.empty else 0
    viability_pct = float(viable_count / total) if total > 0 else 0.0

    if not viable_df.empty:
        viable_df["categoria_exec"] = np.where(viable_df["is_blue_chip"] == True, "BLUE CHIP", "OPORTUNIDADE")
        dist = viable_df["categoria_exec"].value_counts().to_dict()
        dist = {str(k): int(v) for k, v in dist.items()}
    else:
        dist = {}

    opp_df = viable_df[viable_df["is_blue_chip"] == False].copy() if not viable_df.empty else pd.DataFrame()
    if not opp_df.empty:
        opp_df = opp_df.sort_values(["score_total", "avg_fin_volume"], ascending=[False, False])
    top10 = opp_df.head(10).copy() if not opp_df.empty else pd.DataFrame()

    opportunities = []
    for _, a in top10.iterrows():
        opportunities.append(
            {
                "symbol": str(a.get("symbol", "")),
                "tier": str(a.get("tier", "")),
                "score_total": float(a.get("score_total", 0.0) or 0.0),
                "avg_fin_volume": float(a.get("avg_fin_volume", 0.0) or 0.0),
                "volatility_ann": float(a.get("volatility_ann", 0.0) or 0.0),
                "abs_corr_ibov": float(a.get("abs_corr_ibov", 0.0) or 0.0),
                "risk_notes": "Correlação alta com IBOV reduz diversificação; monitorar em dias de stress.",
            }
        )

    risks = [
        "Dependência de dados (MT5 ou Yahoo). Yahoo M15 é limitado (~60 dias).",
        "Market cap pode estar ausente (0) para alguns ativos e reduzir confiança do filtro.",
        "Correlação com IBOV pode concentrar risco sistêmico; considerar balanceamento por setor.",
    ]

    summary = {
        "total_avaliados": total,
        "total_viaveis": viable_count,
        "percentual_viabilidade": viability_pct,
        "distribuicao_por_categorias": dist,
        "principais_oportunidades": opportunities,
        "principais_riscos": risks,
    }

    assets_viaveis = []
    if not viable_df.empty:
        viable_df = viable_df.sort_values(["rank_total", "symbol"])
        thr = (methodology or {}).get("thresholds") or {}
        for _, a in viable_df.iterrows():
            sym = str(a.get("symbol", "")).upper()
            is_blue = bool(a.get("is_blue_chip", False))
            tipo = _infer_asset_type(sym)
            categoria = "BLUE CHIP" if is_blue else "OPORTUNIDADE"
            trade_params = _build_trade_params_for_symbol(sym, is_blue, elite_symbols)

            crit = {
                "liquidez": {
                    "metrica": "avg_fin_volume",
                    "valor": float(a.get("avg_fin_volume", 0.0) or 0.0),
                    "limiar": float(thr.get("liquidity_q50", 0.0) or 0.0),
                    "condicao": ">= limiar",
                    "ok": bool(a.get("crit_liquidez_ok", False)),
                },
                "volatilidade": {
                    "metrica": "volatility_ann",
                    "valor": float(a.get("volatility_ann", 0.0) or 0.0),
                    "limiar_inferior": float(thr.get("volatility_q20", 0.0) or 0.0),
                    "limiar_superior": float(thr.get("volatility_q90", 0.0) or 0.0),
                    "condicao": "entre [limiar_inferior, limiar_superior]",
                    "ok": bool(a.get("crit_vol_ok", False)),
                },
                "correlacao_ibov": {
                    "metrica": "abs_corr_ibov",
                    "valor": float(a.get("abs_corr_ibov", 0.0) or 0.0),
                    "limiar": float(thr.get("abs_corr_q75", 0.0) or 0.0),
                    "condicao": "<= limiar",
                    "ok": bool(a.get("crit_corr_ok", False)),
                },
                "market_cap": {
                    "metrica": "market_cap",
                    "valor": float(a.get("market_cap", 0.0) or 0.0),
                    "limiar": float(thr.get("market_cap_q30", 0.0) or 0.0),
                    "condicao": ">= limiar (quando limiar > 0)",
                    "ok": bool(a.get("crit_mcap_ok", False)),
                },
                "regra_final": {
                    "blue_chip": is_blue,
                    "criteria_passed": int(a.get("criteria_passed", 0) or 0),
                    "liquidez_obrigatoria": True,
                    "min_criterios": 3,
                },
            }

            resultados = {
                "rank_total": int(a.get("rank_total", 0) or 0),
                "tier": str(a.get("tier", "")),
                "score_total": float(a.get("score_total", 0.0) or 0.0),
                "liquidez_avg_fin_volume": float(a.get("avg_fin_volume", 0.0) or 0.0),
                "volatilidade_anualizada": float(a.get("volatility_ann", 0.0) or 0.0),
                "correlacao_ibov": float(a.get("corr_ibov", 0.0) or 0.0),
                "market_cap": float(a.get("market_cap", 0.0) or 0.0),
                "fonte_dados": str(a.get("data_source", "")),
            }

            recomendacoes = {
                "acao_recomendada": "Adicionar ao universo de execução da próxima semana",
                "modo": "BLUE CHIP" if is_blue else "OPORTUNIDADE",
                "parametrizacao": "Usar ELITE_SYMBOLS (se existir) ou DEFAULT_PARAMS como base e refinar via otimização WFO.",
                "monitoramento": "Monitorar volatilidade e correlação diária; ajustar exposição se |corr| subir.",
            }

            prazos = {
                "prazo_estimado": "1 dia útil",
                "recursos_necessarios": ["MT5 conectado", "dados D1/M15", "cache Yahoo habilitado", "tempo de backtest"],
            }

            restricoes = {
                "dependencias": ["MetaTrader5", "yfinance"],
                "observacoes": [
                    "Se o MT5 não tiver o símbolo, o script usa Yahoo como fallback.",
                    "Backtest M15 com Yahoo é limitado e deve ser validado com dados do broker quando possível.",
                ],
            }

            assets_viaveis.append(
                {
                    "identificacao": {"nome": sym, "codigo": sym, "codigo_unico": sym, "tipo": tipo, "categoria": categoria},
                    "ativo": trade_params,
                    "criterios_viabilidade": crit,
                    "resultados": resultados,
                    "recomendacoes": recomendacoes,
                    "prazos_e_recursos": prazos,
                    "restricoes_dependencias": restricoes,
                    "justificativa_tecnica": str(a.get("viability_reason", "")),
                }
            )

    return {
        "metadados": {"timestamp": ts_iso, "versao_sistema": "XP3 PRO v7.0"},
        "sumario_executivo": summary,
        "ativos_viaveis": assets_viaveis,
    }


def update_config_py_with_report(config_py_path: str, report_payload: dict[str, Any]) -> str:
    json_text = json.dumps(report_payload, ensure_ascii=False, indent=2)
    json.loads(json_text)

    json_block = (
        'ATIVOS_VIAVEIS_REPORT_JSON = r"""' + json_text + '"""\n'
        "ATIVOS_VIAVEIS_REPORT = json.loads(ATIVOS_VIAVEIS_REPORT_JSON)\n"
    )

    p = Path(config_py_path)
    original = p.read_text(encoding="utf-8")

    start = "ATIVOS_VIAVEIS_REPORT_JSON = r\"\"\""
    if start in original:
        prefix, rest = original.split(start, 1)
        suffix = rest.split("\"\"\"", 1)
        if len(suffix) != 2:
            raise RuntimeError("Bloco ATIVOS_VIAVEIS_REPORT_JSON corrompido em config.py")
        tail = suffix[1]
        after_line = tail.splitlines(True)
        if after_line and after_line[0].lstrip().startswith("ATIVOS_VIAVEIS_REPORT"):
            after_line = after_line[1:]
        updated = prefix + json_block + "".join(after_line)
    else:
        if not original.endswith("\n"):
            original += "\n"
        updated = original + "\n" + json_block

    p.write_text(updated, encoding="utf-8")
    return str(p)


def write_analysis_files(out_dir: str, ranking_df: pd.DataFrame) -> dict[str, str]:
    out_dir = _ensure_dir(out_dir)
    ts_iso = datetime.now().isoformat()

    evaluated_df, methodology = evaluate_viability(ranking_df, BLUE_CHIPS)

    complete_json_path = os.path.join(out_dir, "analise_completa.json")
    complete_txt_path = os.path.join(out_dir, "analise_completa.txt")
    filtered_json_path = os.path.join(out_dir, "melhores_ativos.json")
    filtered_txt_path = os.path.join(out_dir, "melhores_ativos.txt")

    assets_all = evaluated_df.sort_values(["rank_total", "symbol"]).to_dict(orient="records") if not evaluated_df.empty else []

    viable_df = evaluated_df[evaluated_df["viable"] == True].copy() if not evaluated_df.empty else pd.DataFrame()
    viable_assets = viable_df.sort_values(["rank_total", "symbol"]).to_dict(orient="records") if not viable_df.empty else []

    opp_df = viable_df[viable_df["is_blue_chip"] == False].copy() if not viable_df.empty else pd.DataFrame()
    if not opp_df.empty:
        opp_df = opp_df.sort_values(["score_total", "avg_fin_volume"], ascending=[False, False])
    top10 = opp_df.head(10).to_dict(orient="records") if not opp_df.empty else []

    blue_df = viable_df[viable_df["is_blue_chip"] == True].copy() if not viable_df.empty else pd.DataFrame()
    blue_included = blue_df.sort_values(["symbol"]).to_dict(orient="records") if not blue_df.empty else []

    complete_payload = {
        "generated_at": ts_iso,
        "methodology": methodology,
        "blue_chips": BLUE_CHIPS,
        "assets": assets_all,
    }

    filtered_payload = {
        "generated_at": ts_iso,
        "methodology": methodology,
        "ordering": {
            "viable_assets": "rank_total asc (menor=melhor) para apresentação",
            "top_10_opportunities_excluding_blue_chips": "score_total desc; desempate avg_fin_volume desc",
            "blue_chips_policy": "blue chips sempre viáveis e incluídas",
        },
        "blue_chips": BLUE_CHIPS,
        "blue_chips_included": blue_included,
        "viable_assets": viable_assets,
        "top_10_opportunities_excluding_blue_chips": top10,
    }

    Path(complete_json_path).write_text(json.dumps(complete_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(filtered_json_path).write_text(json.dumps(filtered_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_txt(path: str, title: str, body_lines: list[str]):
        text = "\n".join([title, f"Timestamp: {ts_iso}", "", *body_lines, ""])
        Path(path).write_text(text, encoding="utf-8")

    body_complete = [
        "METODOLOGIA",
        f"- Liquidez: {methodology['ranking_criteria']['liquidity']}",
        f"- Volatilidade: {methodology['ranking_criteria']['volatility']}",
        f"- Correlação IBOV: {methodology['ranking_criteria']['corr_ibov']}",
        f"- Market cap: {methodology['ranking_criteria']['market_cap']}",
        f"- Pesos score_total: liquidez {methodology['score_weights']['liquidity']}, vol {methodology['score_weights']['volatility']}, mcap {methodology['score_weights']['market_cap']}, diversificação {methodology['score_weights']['diversification']}",
        f"- Regra de viabilidade: {methodology['viability_rule']['non_bluechips']}",
        "",
        "RESUMO",
        f"- Total analisados: {len(assets_all)}",
        f"- Viáveis: {len(viable_assets)}",
        f"- Blue chips definidas: {', '.join(BLUE_CHIPS)}",
        "",
        "DETALHES POR ATIVO (ordenado por rank_total)",
    ]
    for a in assets_all:
        body_complete.append(
            f"{a.get('rank_total','-'):>3} | {a.get('symbol',''):>6} | tier {a.get('tier','-')} | "
            f"viável={'SIM' if a.get('viable') else 'NÃO'} | "
            f"liq={_format_money_br(a.get('avg_fin_volume',0.0))} | vol={_format_pct(a.get('volatility_ann',0.0))} | "
            f"corr={_format_number(a.get('corr_ibov',0.0),3)} | mcap={_format_money_br(a.get('market_cap',0.0))} | "
            f"src={a.get('data_source','-')}"
        )
        body_complete.append(f"     justificativa: {a.get('viability_reason','')}")

    _write_txt(complete_txt_path, "ANÁLISE COMPLETA - ATIVOS (B3)", body_complete)

    body_filtered = [
        "CRITÉRIOS / ORDENAÇÃO",
        "- Viáveis: blue chips sempre incluídas; demais precisam atender liquidez e 3/4 critérios.",
        "- Top 10 oportunidades (exclui blue chips): score_total desc; desempate por liquidez (avg_fin_volume) desc.",
        "",
        "BLUE CHIPS (sempre viáveis)",
        ", ".join(BLUE_CHIPS),
        "",
        "TOP 10 OPORTUNIDADES (EXCLUINDO BLUE CHIPS)",
    ]
    if top10:
        for i, a in enumerate(top10, start=1):
            body_filtered.append(
                f"{i:>2}. {a.get('symbol','')} | tier {a.get('tier','-')} | score={_format_number(a.get('score_total',0.0),3)} | "
                f"liq={_format_money_br(a.get('avg_fin_volume',0.0))} | vol={_format_pct(a.get('volatility_ann',0.0))} | "
                f"|corr|={_format_number(a.get('abs_corr_ibov',0.0),3)}"
            )
    else:
        body_filtered.append("(nenhuma oportunidade não-bluechip disponível)")

    body_filtered.append("")
    body_filtered.append("ATIVOS VIÁVEIS (LISTA COMPLETA)")
    for a in viable_assets:
        body_filtered.append(
            f"{a.get('rank_total','-'):>3} | {a.get('symbol',''):>6} | tier {a.get('tier','-')} | "
            f"liq={_format_money_br(a.get('avg_fin_volume',0.0))} | vol={_format_pct(a.get('volatility_ann',0.0))} | "
            f"corr={_format_number(a.get('corr_ibov',0.0),3)} | mcap={_format_money_br(a.get('market_cap',0.0))}"
        )

    _write_txt(filtered_txt_path, "MELHORES ATIVOS - PRÓXIMA SEMANA (B3)", body_filtered)

    config_update_status: dict[str, Any] = {"updated": False}
    try:
        config_py_path = os.path.join(os.path.dirname(__file__), "config.py")
        report_payload = build_viable_report_payload(evaluated_df, methodology)
        update_config_py_with_report(config_py_path, report_payload)
        config_update_status = {"updated": True, "path": config_py_path}
    except Exception as e:
        config_update_status = {"updated": False, "error": str(e)}

    return {
        "analise_completa_json": complete_json_path,
        "analise_completa_txt": complete_txt_path,
        "melhores_ativos_json": filtered_json_path,
        "melhores_ativos_txt": filtered_txt_path,
        "config_py_report_status": config_update_status,
    }


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
    return np.vstack(norm).mean(axis=0) * initial


def run_portfolio_size_study(rcfg: RankingConfig, ranking_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    if ranking_df is None or ranking_df.empty:
        return pd.DataFrame(), {}
    if not _mt5_initialize_with_retry():
        raise RuntimeError("Falha ao conectar ao MT5")

    elite_params = _load_elite_params_latest(rcfg.output_dir)
    curves: dict[int, np.ndarray] = {}
    rows: list[dict[str, Any]] = []

    sizes = [int(x) for x in rcfg.portfolio_sizes]
    max_n = max(sizes) if sizes else 0
    universe = ranking_df.head(max_n)["symbol"].tolist()

    per_symbol: dict[str, dict[str, Any]] = {}
    for idx, sym in enumerate(universe, start=1):
        params = dict(DEFAULT_PARAMS)
        ep = elite_params.get(sym)
        if isinstance(ep, dict):
            for k in DEFAULT_PARAMS.keys():
                if k in ep:
                    params[k] = ep.get(k)

        df_m15 = _get_rates(sym, timeframe=mt5.TIMEFRAME_M15, bars=int(rcfg.portfolio_m15_bars), start_pos=0, source="auto", output_dir=rcfg.output_dir)
        if df_m15 is None or df_m15.empty or len(df_m15) < 200:
            continue

        m = backtest_params_on_df(params, df_m15)
        per_symbol[sym] = {
            "equity_curve": np.asarray(m.get("equity_curve", [100000.0]), dtype=float),
            "trades": int(m.get("total_trades", 0) or 0),
            "costs_paid": _safe_float(m.get("costs_paid", 0.0), 0.0),
        }

    for n in sizes:
        selected = universe[: int(n)]
        per_eq = []
        total_trades = 0
        costs_paid = 0.0

        for sym in selected:
            item = per_symbol.get(sym)
            if not item:
                continue
            per_eq.append(item["equity_curve"])
            total_trades += int(item["trades"])
            costs_paid += float(item["costs_paid"])

        port_eq = _portfolio_from_equity_curves(per_eq, initial=100000.0)
        curves[int(n)] = port_eq

        m_port = compute_advanced_metrics(port_eq.tolist())
        rows.append(
            {
                "n_assets": int(n),
                "n_assets_effective": int(len(per_eq)),
                "total_return": _safe_float(m_port.get("total_return", 0.0), 0.0),
                "sharpe": _safe_float(m_port.get("sharpe", 0.0), 0.0),
                "sortino": _safe_float(m_port.get("sortino", 0.0), 0.0),
                "calmar": _safe_float(m_port.get("calmar", 0.0), 0.0),
                "max_drawdown": _safe_float(m_port.get("max_drawdown", 1.0), 1.0),
                "final_equity": _safe_float(m_port.get("final_equity", 100000.0), 100000.0),
                "total_trades": int(total_trades),
                "costs_paid_proxy": float(costs_paid),
            }
        )

    return pd.DataFrame(rows).sort_values("n_assets").reset_index(drop=True), curves


def generate_reports(rcfg: RankingConfig, ranking_df: pd.DataFrame, portfolio_df: pd.DataFrame, curves: dict[int, np.ndarray]) -> dict[str, str]:
    out_dir = _ensure_dir(rcfg.output_dir)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    artifacts: dict[str, str] = {}

    ranking_csv = os.path.join(out_dir, f"b3_asset_ranking_{ts}.csv")
    ranking_latest = os.path.join(out_dir, "b3_asset_ranking_latest.csv")
    ranking_df.to_csv(ranking_csv, index=False)
    ranking_df.to_csv(ranking_latest, index=False)
    artifacts["ranking_csv"] = ranking_csv
    artifacts["ranking_latest"] = ranking_latest

    ranking_json = os.path.join(out_dir, f"b3_asset_ranking_{ts}.json")
    ranking_latest_json = os.path.join(out_dir, "b3_asset_ranking_latest.json")
    tiers: dict[str, list[dict[str, Any]]] = {"A": [], "B": [], "C": [], "D": []}
    for t in tiers.keys():
        tdf = ranking_df[ranking_df["tier"] == t] if "tier" in ranking_df.columns else ranking_df
        tiers[t] = tdf.to_dict(orient="records")
    payload = {"generated_at": datetime.now().isoformat(), "tiers": tiers}
    Path(ranking_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(ranking_latest_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    artifacts["ranking_json"] = ranking_json
    artifacts["ranking_latest_json"] = ranking_latest_json

    if portfolio_df is not None and not portfolio_df.empty:
        portfolio_csv = os.path.join(out_dir, f"portfolio_size_study_{ts}.csv")
        portfolio_latest = os.path.join(out_dir, "portfolio_size_study_latest.csv")
        portfolio_df.to_csv(portfolio_csv, index=False)
        portfolio_df.to_csv(portfolio_latest, index=False)
        artifacts["portfolio_csv"] = portfolio_csv
        artifacts["portfolio_latest"] = portfolio_latest

    if ranking_df is not None and not ranking_df.empty:
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
        rank_latest = os.path.join(out_dir, "b3_asset_ranking_chart_latest.html")
        fig_rank.write_html(rank_html, include_plotlyjs="cdn")
        fig_rank.write_html(rank_latest, include_plotlyjs="cdn")
        artifacts["ranking_chart_html"] = rank_html
        artifacts["ranking_chart_latest"] = rank_latest

    if portfolio_df is not None and not portfolio_df.empty and curves:
        fig_eq = go.Figure()
        for n, eq in sorted(curves.items(), key=lambda kv: kv[0]):
            fig_eq.add_trace(go.Scatter(y=eq, mode="lines", name=f"{n} ativos"))
        fig_eq.update_layout(
            title="Backtest comparativo por tamanho de carteira (equity agregada)",
            xaxis_title="Barra (M15)",
            yaxis_title="Equity (R$)",
            template="plotly_white",
            height=600,
        )
        eq_html = os.path.join(out_dir, f"portfolio_size_equity_{ts}.html")
        eq_latest = os.path.join(out_dir, "portfolio_size_equity_latest.html")
        fig_eq.write_html(eq_html, include_plotlyjs="cdn")
        fig_eq.write_html(eq_latest, include_plotlyjs="cdn")
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
        fig_metrics.write_html(met_html, include_plotlyjs="cdn")
        fig_metrics.write_html(met_latest, include_plotlyjs="cdn")
        artifacts["portfolio_metrics_html"] = met_html
        artifacts["portfolio_metrics_latest"] = met_latest

    artifacts.update(write_analysis_files(out_dir, ranking_df))

    return artifacts


def recommend_portfolio_size(portfolio_df: pd.DataFrame) -> dict[str, Any]:
    if portfolio_df is None or portfolio_df.empty:
        return {"recommended_n": None, "reason": "Sem dados"}
    df = portfolio_df.copy()
    df["score"] = df["sharpe"].fillna(0.0) - 1.25 * df["max_drawdown"].fillna(1.0) + 0.15 * df["calmar"].fillna(0.0)
    best = df.sort_values(["score", "sharpe"], ascending=False).iloc[0]
    return {
        "recommended_n": int(best["n_assets"]),
        "n_assets_effective": int(best.get("n_assets_effective", 0) or 0),
        "sharpe": float(best.get("sharpe", 0.0) or 0.0),
        "max_drawdown": float(best.get("max_drawdown", 1.0) or 1.0),
        "calmar": float(best.get("calmar", 0.0) or 0.0),
        "reason": "Maximiza score (Sharpe - 1.25*DD + 0.15*Calmar) com custos embutidos no backtest",
    }


# ======================================================================================
# FLUXO PRINCIPAL (CLI)
# ======================================================================================

def run_once(args) -> dict[str, Any]:
    rcfg = RankingConfig(d1_window_days=int(args.d1_window_days), portfolio_m15_bars=int(args.portfolio_bars), output_dir=str(args.output_dir))
    _ensure_dir(rcfg.output_dir)

    t0 = time.time()
    print("=" * 90, flush=True)
    print("📊 B3 Weekly Rankings (single-file)", flush=True)
    print(f"🕒 Início: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"⚙️ Mode: {args.mode} | D1 window: {rcfg.d1_window_days} | Portfolio bars (M15): {rcfg.portfolio_m15_bars}", flush=True)
    print(f"📁 Saída: {os.path.abspath(rcfg.output_dir)}", flush=True)
    print("=" * 90, flush=True)

    ranking_df = build_asset_ranking(rcfg)
    print(f"✅ Ranking gerado: {len(ranking_df)} ativos", flush=True)
    portfolio_df = pd.DataFrame()
    curves: dict[int, np.ndarray] = {}

    if args.mode in ("portfolio", "full"):
        print("🧪 Rodando estudo de tamanho de carteira (5/10/15/20)...", flush=True)
        portfolio_df, curves = run_portfolio_size_study(rcfg, ranking_df)
        if portfolio_df is not None and not portfolio_df.empty:
            print("✅ Estudo de carteiras concluído:", flush=True)
            for _, row in portfolio_df.iterrows():
                n = int(row.get("n_assets", 0) or 0)
                eff = int(row.get("n_assets_effective", 0) or 0)
                sharpe = float(row.get("sharpe", 0.0) or 0.0)
                dd = float(row.get("max_drawdown", 1.0) or 1.0)
                tr = float(row.get("total_return", 0.0) or 0.0)
                print(f" - {n} ativos (efetivo {eff}): Sharpe={sharpe:.3f} | MaxDD={dd:.2%} | Ret={tr:.2%}", flush=True)

    artifacts = generate_reports(rcfg, ranking_df, portfolio_df, curves)
    print("🧾 Artefatos gerados:", flush=True)
    for k, v in sorted(artifacts.items(), key=lambda kv: kv[0]):
        if v:
            print(f" - {k}: {v}", flush=True)

    top20_payload = None
    if args.mode in ("top20", "full"):
        print("🔎 Verificando parâmetros no MT5 (top20)...", flush=True)
        top20_payload = build_top20_mt5_verification(rcfg, ranking_df, mt5_bars=int(args.mt5_verify_bars))
        top20_artifacts = _write_top20_files(rcfg.output_dir, top20_payload)
        artifacts.update(top20_artifacts)
        for k, v in sorted(top20_artifacts.items(), key=lambda kv: kv[0]):
            print(f" - {k}: {v}", flush=True)

    rec = {}
    if args.mode in ("portfolio", "full"):
        rec = recommend_portfolio_size(portfolio_df)
        Path(os.path.join(rcfg.output_dir, "portfolio_size_recommendation_latest.json")).write_text(
            json.dumps({"generated_at": datetime.now().isoformat(), "recommendation": rec}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if rec.get("recommended_n") is not None:
            print(
                f"🏁 Recomendação: {rec.get('recommended_n')} ativos "
                f"(Sharpe {float(rec.get('sharpe', 0.0) or 0.0):.3f} | "
                f"MaxDD {float(rec.get('max_drawdown', 1.0) or 1.0):.2%} | "
                f"Calmar {float(rec.get('calmar', 0.0) or 0.0):.3f})",
                flush=True,
            )

    dt = time.time() - t0
    print(f"⏱️ Tempo total: {dt:.1f}s", flush=True)
    print("=" * 90, flush=True)

    return {"artifacts": artifacts, "recommendation": rec, "top20": top20_payload}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["rank", "portfolio", "top20", "full"], default="full")
    parser.add_argument("--schedule", action="store_true")
    parser.add_argument("--d1-window-days", type=int, default=126)
    parser.add_argument("--portfolio-bars", type=int, default=1200)
    parser.add_argument("--mt5-verify-bars", type=int, default=1800)
    parser.add_argument("--output-dir", type=str, default=OPTIMIZER_OUTPUT)
    args = parser.parse_args()

    try:
        if args.schedule:
            try:
                import schedule
            except Exception:
                run_once(args)
                return 0

            schedule.every().sunday.at("22:00").do(lambda: run_once(args))
            while True:
                schedule.run_pending()
                time.sleep(30)
        else:
            run_once(args)
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as e:
        print(f"ERRO: {e}")
        return 1
    finally:
        try:
            mt5.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
