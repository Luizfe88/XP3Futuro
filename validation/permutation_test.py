"""Permutation Test utilities for XP3Futuro

This module implements a Monte Carlo permutation test with optional block
permutation to preserve autocorrelation (key for B3 futures returns).

The primary entry-point is `run_permutation_test` which loads a JSON trade
history, computes the chosen metric, shuffles returns, and generates a
report (PNG histogram + HTML/PDF summary) under `validation/reports/`.

The function returns a dictionary containing the original value, p-value,
paths to the artifacts and the raw permutation array for further analysis.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np


# ------------------------------------------------------------------
# internal helpers
# ------------------------------------------------------------------


def _compute_metric(returns: np.ndarray, metric: str) -> float:
    """Compute one of the supported metrics on a return series."""
    returns = np.asarray(returns, dtype=float)
    if metric == "profit_factor":
        profit = returns[returns > 0].sum()
        loss = -returns[returns < 0].sum()
        if loss <= 0:
            return float("inf") if profit > 0 else 0.0
        return profit / loss
    elif metric == "net_profit":
        return returns.sum()
    elif metric == "sharpe_ratio":
        std = returns.std()
        if std == 0:
            return 0.0
        # we omit annualization factor since p-value is invariant to it
        return returns.mean() / std
    else:
        raise ValueError(f"Métrica desconhecida: {metric}")


def _block_permute(arr: np.ndarray, block_size: int) -> np.ndarray:
    """Permute the array by shuffling contiguous blocks of size `block_size`."""
    if block_size <= 1 or block_size >= len(arr):
        return np.random.permutation(arr)
    n = len(arr)
    blocks = [arr[i : i + block_size] for i in range(0, n, block_size)]
    np.random.shuffle(blocks)
    return np.concatenate(blocks)


# ------------------------------------------------------------------
# public API
# ------------------------------------------------------------------


def run_permutation_test(
    trade_history_path: str = "ml_trade_history.json",
    n_permutations: int = 5000,
    metric: str = "profit_factor",
    use_block_permutation: bool = True,
    block_size: int = 3,
    bootstrap: bool = True,
) -> Dict[str, Any]:
    """Execute a permutation test on a series of trade returns.

    Parameters
    ----------
    trade_history_path : str
        Path to a JSON file containing a list of trade records. Each record
        should contain at least one of the keys ``profit``, ``pnl`` or
        ``pnl_pct``.
    n_permutations : int
        Number of Monte Carlo samples to generate.
    use_block_permutation : bool
        When ``True`` the algorithm will shuffle blocks of `block_size`
        elements instead of individual trades. This preserves short-range
        autocorrelation typical in futures returns.
    block_size : int
        Size of the blocks when ``use_block_permutation`` is enabled.
    bootstrap : bool
        If ``True`` samples are drawn **with replacement** (bootstrapping);
        otherwise returns are permuted without replacement.  Bootstrapping is
        the preferred method for profit metrics since individual trades can
        repeat, producing realistic alternative equity curves.
    metric : str
        Name of the metric to compute (``profit_factor``, ``sharpe_ratio``
        or ``net_profit``).
    use_block_permutation : bool
        When ``True`` the algorithm will shuffle blocks of `block_size`
        elements instead of individual trades. This preserves short-range
        autocorrelation typical in futures returns.
    block_size : int
        Size of the blocks when ``use_block_permutation`` is enabled.

    Returns
    -------
    dict
        Information about the test including ``original``, ``p_value``,
        ``plot`` path and report paths.
    """
    # load history
    if not os.path.exists(trade_history_path):
        raise FileNotFoundError(
            f"Arquivo de histórico não encontrado: {trade_history_path}"
        )

    with open(trade_history_path, "r", encoding="utf-8") as f:
        data = json.load(f) or []

    # extract returns
    returns: List[float] = []
    for rec in data:
        if not isinstance(rec, dict):
            continue
        for key in ("profit", "pnl", "pnl_pct"):
            if key in rec:
                try:
                    returns.append(float(rec[key]))
                except Exception:
                    pass
                break
    if len(returns) == 0:
        raise ValueError(
            "Nenhuma variável de retorno encontrada no histórico de trades"
        )

    arr = np.array(returns, dtype=float)
    orig_val = _compute_metric(arr, metric)

    perm_vals = []
    for _ in range(n_permutations):
        if bootstrap:
            # draw with replacement
            if use_block_permutation and block_size > 1:
                # block bootstrap: sample blocks with replacement and
                # concatenate (trim to original length if necessary)
                blocks = [
                    arr[i : i + block_size] for i in range(0, len(arr), block_size)
                ]
                chosen = np.random.choice(blocks, size=len(blocks), replace=True)
                perm = np.concatenate(chosen)[: len(arr)]
            else:
                perm = np.random.choice(arr, size=len(arr), replace=True)
        else:
            # traditional permutation
            if use_block_permutation:
                perm = _block_permute(arr, block_size)
            else:
                perm = np.random.permutation(arr)
        perm_vals.append(_compute_metric(perm, metric))
    perm_vals = np.array(perm_vals)

    # one-sided p-value (larger is better)
    p_value = (np.sum(perm_vals >= orig_val) + 1) / (n_permutations + 1)

    # prepare output directory
    outdir = os.path.join("validation", "reports")
    os.makedirs(outdir, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(outdir, f"permutation_{metric}_{ts}")
    plot_path = base + ".png"
    html_path = base + ".html"
    pdf_path = base + ".pdf"

    # histogram
    plt.figure(figsize=(8, 5))
    plt.hist(perm_vals, bins=50, alpha=0.7, color="#4C72B0")
    plt.axvline(orig_val, color="red", linestyle="--", label=f"original={orig_val:.4f}")
    plt.title(f"Permutation test ({metric})\n p-value={p_value:.4f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    # html report
    with open(html_path, "w", encoding="utf-8") as f:
        f.write("<html><body>")
        f.write(f"<h1>Permutation Test Report ({metric})</h1>")
        f.write(f"<p>original: {orig_val:.6f}</p>")
        f.write(f"<p>p-value: {p_value:.6f}</p>")
        f.write(f'<img src="{os.path.basename(plot_path)}" style="max-width:100%"/>')
        f.write("</body></html>")

    # simple pdf output (same figure)
    try:
        from matplotlib.backends.backend_pdf import PdfPages

        fig = plt.figure(figsize=(8, 5))
        plt.hist(perm_vals, bins=50, alpha=0.7, color="#4C72B0")
        plt.axvline(orig_val, color="red", linestyle="--")
        plt.title(f"Permutation test ({metric})\n p-value={p_value:.4f}")
        plt.tight_layout()
        with PdfPages(pdf_path) as pdf:
            pdf.savefig(fig)
        plt.close(fig)
    except Exception:
        # PDF generation is nice-to-have; failures should not break the test
        pass

    return {
        "metric": metric,
        "original": orig_val,
        "p_value": p_value,
        "permuted": perm_vals,
        "plot": plot_path,
        "html_report": html_path,
        "pdf_report": pdf_path,
    }
