import os
import json
import logging
import argparse
from typing import List, Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

# try tqdm and colorama, fallback to simple prints if not installed
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except Exception:
    TQDM_AVAILABLE = False

try:
    from colorama import init as colorama_init, Fore, Style
    colorama_init(autoreset=True)
    C_GREEN = Fore.GREEN; C_RED = Fore.RED; C_YELLOW = Fore.YELLOW; C_CYAN = Fore.CYAN; C_RESET = Style.RESET_ALL; C_BOLD = Style.BRIGHT
except Exception:
    C_GREEN = ""; C_RED = ""; C_YELLOW = ""; C_CYAN = ""; C_RESET = ""; C_BOLD = ""

# try to import optimizer module (two possible names)
try:
    import optimizer_updated as base
except Exception:
    try:
        import optimizer as base
    except Exception:
        base = None

import config
import re
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("optimizer_sector_weekly_turbo")

def load_all_sector_symbols() -> List[str]:
    secmap = getattr(config, "SECTOR_MAP", {}) or {}
    # Your SECTOR_MAP maps "TICKER" -> "SECTOR". We extract keys.
    all_syms = set()
    for k in secmap.keys():
        if isinstance(k, str) and k.strip():
            all_syms.add(k.upper().strip())
    return sorted(all_syms)

def _optimize_worker(sym: str, bars: int, maxevals: int) -> Dict[str, Any]:
    """
    Worker wrapper to call base.optimize_symbol_robust for a single symbol.
    Returns a dict with symbol and result status.
    """
    try:
        if base is None:
            return {"symbol": sym, "error": "no_base_module"}
        res = base.optimize_symbol_robust(sym, base_dir=getattr(base, "OPT_OUTPUT_DIR", "optimizer_output"), max_evals=maxevals)
        return {"symbol": sym, "result": res}
    except Exception as e:
        return {"symbol": sym, "error": str(e)}

def run_parallel_robust(symbols: List[str], bars: int, maxevals: int, workers: int):
    os.makedirs(getattr(base, "OPT_OUTPUT_DIR", "optimizer_output") if base else "optimizer_output", exist_ok=True)
    results = {}
    total = len(symbols)
    if total == 0:
        logger.info("No symbols to optimize.")
        return results

    logger.info(f"{C_CYAN}Starting parallel robust optimization: {total} symbols, workers={workers}{C_RESET}")
    start_all = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_optimize_worker, s, bars, maxevals): s for s in symbols}
        if TQDM_AVAILABLE:
            pbar = tqdm(total=total, desc="Robust Opt", unit="sym")
        else:
            pbar = None
        for fut in as_completed(futures):
            sym = futures[fut]
            try:
                r = fut.result()
                results[sym] = r
                if r.get("error"):
                    logger.warning(f"{C_YELLOW}OPT {sym} error: {r.get('error')}{C_RESET}")
                else:
                    logger.info(f"{C_GREEN}OPT {sym} done{C_RESET}")
            except Exception as e:
                results[sym] = {"symbol": sym, "error": str(e)}
                logger.exception(f"{C_RED}OPT {sym} failed{C_RESET}")
            if pbar:
                pbar.update(1)
        if pbar:
            pbar.close()
    elapsed = time.time() - start_all
    logger.info(f"{C_CYAN}Parallel robust optimization finished in {elapsed:.1f}s{C_RESET}")
    return results

def run_serial_ml(symbols: List[str], bars: int):
    logger.info(f"{C_CYAN}Starting ML training in series for {len(symbols)} symbols{C_RESET}")
    ml_results = {}
    for sym in symbols:
        try:
            df = base.load_historical_bars(sym, bars=bars)
            if df is None:
                logger.warning(f"{C_YELLOW}ML {sym}: no data, skipped{C_RESET}")
                ml_results[sym] = {"skipped": True}
                continue
            logger.info(f"{C_CYAN}ML training {sym}...{C_RESET}")
            out = base.train_ml_model(sym, df, base_dir=getattr(base, "OPT_OUTPUT_DIR", "optimizer_output"))
            ml_results[sym] = out or {"status": "no_model"}
        except Exception as e:
            logger.exception(f"{C_RED}ML {sym} failed: {e}{C_RESET}")
            ml_results[sym] = {"error": str(e)}
    logger.info(f"{C_CYAN}ML training finished{C_RESET}")
    return ml_results

def safe_update_config_optimized(updated: dict):
    """
    Insert or replace OPTIMIZED_PARAMS block in config.py with updated dict.
    """
    cfg_path = "config.py"
    try:
        if not os.path.exists(cfg_path):
            logger.error(f"config.py not found at {os.path.abspath(cfg_path)}")
            return False
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg_text = f.read()

        new_block = "OPTIMIZED_PARAMS = " + json.dumps(updated, indent=2, ensure_ascii=False)

        if "OPTIMIZED_PARAMS" in cfg_text:
            cfg_text = re.sub(r"OPTIMIZED_PARAMS\s*=\s*\{[\s\S]*?\}", new_block, cfg_text, flags=re.DOTALL)
        else:
            cfg_text += "\n\n" + new_block

        # backup original
        bak = cfg_path + ".bak"
        with open(bak, "w", encoding="utf-8") as f:
            f.write(cfg_text if False else open(cfg_path, "r", encoding="utf-8").read())

        with open(cfg_path, "w", encoding="utf-8") as f:
            f.write(cfg_text)
        return True
    except Exception as e:
        logger.exception(f"Failed updating config.py: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Weekly Optimizer Turbo (Sector Map keys)")
    parser.add_argument('--bars', type=int, default=4000)
    parser.add_argument('--mode', type=str, default='robust', choices=['robust','ml','both'])
    parser.add_argument('--maxevals', type=int, default=300)
    parser.add_argument('--workers', type=int, default=0, help="Number of parallel workers (0 = auto)")
    args = parser.parse_args()

    symbols = load_all_sector_symbols()
    logger.info(f"Weekly Optimizer (turbo) loading {len(symbols)} symbols from SECTOR_MAP")

    if not symbols:
        logger.warning("No symbols found in SECTOR_MAP. Exiting.")
        return

    # Prepare workers: auto-detect but conservative for low-end CPU (i3)
    cpu = os.cpu_count() or 1
    if args.workers and args.workers > 0:
        workers = max(1, args.workers)
    else:
        # keep some margin: min(cpu-1, 3) but at least 1
        workers = max(1, min(max(1, cpu-1), 3))

    # liquidity filter
    MIN_VOL = getattr(config, "MIN_WEEKLY_OPT_VOL", 500)
    filtered = []
    skipped = []
    for s in symbols:
        df_check = base.load_historical_bars(s, bars=300)
        if df_check is None or df_check.empty:
            logger.warning(f"{C_YELLOW}{s} - no data, skipped{C_RESET}")
            skipped.append(s)
            continue
        vol_col = "tick_volume" if "tick_volume" in df_check.columns else ("volume" if "volume" in df_check.columns else None)
        if vol_col:
            vol_med = df_check[vol_col].tail(50).mean()
            if vol_med < MIN_VOL:
                logger.warning(f"{C_YELLOW}{s} - low volume ({vol_med:.1f}) skipped{C_RESET}")
                skipped.append(s)
                continue
        filtered.append(s)

    logger.info(f"Symbols to optimize: {len(filtered)} (skipped {len(skipped)})")

    # parallel robust optimization
    robust_results = run_parallel_robust(filtered, bars=args.bars, maxevals=args.maxevals, workers=workers)

    # optionally run ML in series
    ml_results = {}
    if args.mode in ('ml','both'):
        ml_results = run_serial_ml(filtered, bars=args.bars)

    # collect best params into dict for config update
    optimized = {}
    out_dir = getattr(base, "OPT_OUTPUT_DIR", "optimizer_output") if base else "optimizer_output"
    for s in filtered:
        fp = os.path.join(out_dir, f"{s}.json")
        if os.path.exists(fp):
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    optimized[s] = json.load(f)
            except Exception:
                logger.exception(f"Failed reading {fp}")

    if optimized:
        ok = safe_update_config_optimized(optimized)
        if ok:
            logger.info(f"{C_GREEN}config.py updated with OPTIMIZED_PARAMS block{C_RESET}")
        else:
            logger.warning(f"{C_YELLOW}config.py update failed{C_RESET}")

    logger.info("Weekly optimizer (turbo) finished.")

if __name__ == "__main__":
    main()
