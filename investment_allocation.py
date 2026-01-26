import json
import os
import logging
from typing import List, Dict, Any
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("investment_allocation")

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def validate_config(cfg: Dict[str, Any]) -> None:
    if not isinstance(cfg.get("category"), str) or not cfg["category"]:
        raise ValueError("category inválido")
    if not isinstance(cfg.get("weight"), (int, float)) or cfg["weight"] <= 0:
        raise ValueError("weight inválido")
    if not isinstance(cfg.get("ema_short"), int) or cfg["ema_short"] <= 0:
        raise ValueError("ema_short inválido")
    if not isinstance(cfg.get("ema_long"), int) or cfg["ema_long"] <= 0:
        raise ValueError("ema_long inválido")
    if cfg["ema_short"] >= cfg["ema_long"]:
        raise ValueError("ema_short deve ser menor que ema_long")
    if not isinstance(cfg.get("rsi_low"), (int, float)):
        raise ValueError("rsi_low inválido")
    if not isinstance(cfg.get("rsi_high"), (int, float)):
        raise ValueError("rsi_high inválido")
    if cfg["rsi_low"] >= cfg["rsi_high"]:
        raise ValueError("rsi_low deve ser menor que rsi_high")
    if not isinstance(cfg.get("adx_threshold"), (int, float)) or cfg["adx_threshold"] <= 0:
        raise ValueError("adx_threshold inválido")
    if not isinstance(cfg.get("mom_min"), (int, float)):
        raise ValueError("mom_min inválido")
    if not isinstance(cfg.get("sl_atr_multiplier"), (int, float)) or cfg["sl_atr_multiplier"] <= 0:
        raise ValueError("sl_atr_multiplier inválido")
    if not isinstance(cfg.get("tp_mult"), (int, float)) or cfg["tp_mult"] <= 0:
        raise ValueError("tp_mult inválido")

def compute_signals(asset: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    ema_convergence = float(asset["ema_short"]) > float(asset["ema_long"])
    rsi_range = float(cfg["rsi_low"]) <= float(asset["rsi"]) <= float(cfg["rsi_high"])
    adx_ok = float(asset["adx"]) >= float(cfg["adx_threshold"])
    mom_ok = float(asset["mom"]) >= float(cfg["mom_min"])
    return {
        "ema_convergence": ema_convergence,
        "rsi_range": rsi_range,
        "adx_ok": adx_ok,
        "mom_ok": mom_ok
    }

def compute_weight(asset: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    signals = compute_signals(asset, cfg)
    qualified = all([signals["ema_convergence"], signals["rsi_range"], signals["adx_ok"], signals["mom_ok"]])
    if not qualified:
        return {"symbol": asset["symbol"], "qualified": False, "raw_weight": 0.0, "signals": signals, "risk_factor": 1.0}
    base = float(cfg["weight"])
    score = sum([1 if v else 0 for v in signals.values()])
    signal_factor = 1.0 + 0.25 * max(0, score - 1)
    asset_sl = float(asset.get("sl_atr", cfg["sl_atr_multiplier"]))
    risk_factor = max(0.5, asset_sl / float(cfg["sl_atr_multiplier"]))
    raw_weight = base * signal_factor / risk_factor
    return {"symbol": asset["symbol"], "qualified": True, "raw_weight": raw_weight, "signals": signals, "risk_factor": risk_factor}

def allocate_assets(assets: List[Dict[str, Any]], cfg: Dict[str, Any]) -> Dict[str, Any]:
    validate_config(cfg)
    results = [compute_weight(a, cfg) for a in assets]
    qualified = [r for r in results if r["qualified"] and r["raw_weight"] > 0]
    total_raw = float(sum(r["raw_weight"] for r in qualified))
    if total_raw <= 0:
        return {"allocations": [], "total_percent": 0.0, "details": results}
    allocations = []
    for r in qualified:
        pct = (r["raw_weight"] / total_raw) * 100.0
        allocations.append({"symbol": r["symbol"], "percent": pct, "signals": r["signals"], "risk_factor": r["risk_factor"]})
    total_percent = float(sum(a["percent"] for a in allocations))
    if total_percent > 100.0:
        scale = 100.0 / total_percent
        for a in allocations:
            a["percent"] = a["percent"] * scale
        total_percent = 100.0
    return {"allocations": allocations, "total_percent": total_percent, "details": results}

def write_report(output_dir: str, cfg: Dict[str, Any], allocation: Dict[str, Any]) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "allocation_report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("CONFIG\n")
        f.write(json.dumps(cfg, indent=2))
        f.write("\n\nDETALHES POR ATIVO\n")
        for r in allocation["details"]:
            f.write(f"{r['symbol']}: qualified={r['qualified']}, raw_weight={r['raw_weight']:.6f}, risk_factor={r['risk_factor']:.4f}\n")
            s = r["signals"]
            f.write(f"  signals: ema_convergence={s['ema_convergence']}, rsi_range={s['rsi_range']}, adx_ok={s['adx_ok']}, mom_ok={s['mom_ok']}\n")
        f.write("\nALOCACAO FINAL\n")
        for a in allocation["allocations"]:
            f.write(f"{a['symbol']}: {a['percent']:.2f}%\n")
        f.write(f"\nTOTAL: {allocation['total_percent']:.2f}%\n")
    logger.info(f"Relatório salvo em {path}")
    return path

def run(assets: List[Dict[str, Any]], config_path: str = "investment_config.json", output_dir: str = "allocation_output") -> Dict[str, Any]:
    cfg = load_config(config_path)
    allocation = allocate_assets(assets, cfg)
    write_report(output_dir, cfg, allocation)
    return allocation

if __name__ == "__main__":
    sample_assets = [
        {"symbol": "VALE3", "ema_short": 105.0, "ema_long": 100.0, "rsi": 50.0, "adx": 32.0, "mom": 0.02, "sl_atr": 3.0},
        {"symbol": "PETR4", "ema_short": 90.0, "ema_long": 100.0, "rsi": 45.0, "adx": 20.0, "mom": -0.01, "sl_atr": 3.5},
        {"symbol": "ITUB4", "ema_short": 110.0, "ema_long": 100.0, "rsi": 40.0, "adx": 30.0, "mom": 0.01, "sl_atr": 2.8}
    ]
    result = run(sample_assets)
    logger.info(json.dumps(result, indent=2))
