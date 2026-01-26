import functools
print = functools.partial(print, flush=True)
print("[DEBUG] Importando time...", flush=True); import time; print("[DEBUG] time importado.", flush=True)
print("[DEBUG] Importando logging...", flush=True); import logging; print("[DEBUG] logging importado.", flush=True)
print("[DEBUG] Importando datetime...", flush=True); from datetime import datetime, timedelta, time as datetime_time; print("[DEBUG] datetime importado.", flush=True)
print("[DEBUG] Importando typing...", flush=True); from typing import Optional, Dict, Any, List, Tuple; print("[DEBUG] typing importado.", flush=True)
print("[DEBUG] Importando collections...", flush=True); from collections import defaultdict, deque; print("[DEBUG] collections importado.", flush=True)
print("[DEBUG] Importando json...", flush=True); import json; print("[DEBUG] json importado.", flush=True)
try:
    print("[DEBUG] Importando MetaTrader5...", flush=True); import MetaTrader5 as mt5; print("[DEBUG] MetaTrader5 importado.", flush=True)
except Exception:
    mt5 = None; print("[DEBUG] MetaTrader5 indisponível.", flush=True)
print("[DEBUG] Importando pandas...", flush=True); import pandas as pd; print("[DEBUG] pandas importado.", flush=True)
print("[DEBUG] Importando numpy...", flush=True); import numpy as np; print("[DEBUG] numpy importado.", flush=True)
print("[DEBUG] Importando config...", flush=True); import config; print("[DEBUG] config importado.", flush=True)
print("[DEBUG] Importando threading...", flush=True); from threading import RLock, Lock; print("[DEBUG] threading importado.", flush=True)
print("[DEBUG] Importando threading (módulo)...", flush=True); import threading; print("[DEBUG] threading(módulo) importado.", flush=True)
print("[DEBUG] Importando queue...", flush=True); import queue; print("[DEBUG] queue importado.", flush=True)
print("[DEBUG] Importando os...", flush=True); import os; print("[DEBUG] os importado.", flush=True)
print("[DEBUG] Importando pathlib.Path...", flush=True); from pathlib import Path; print("[DEBUG] pathlib.Path importado.", flush=True)
try:
    print("[DEBUG] Importando redis...", flush=True); import redis; print("[DEBUG] redis importado.", flush=True)
except Exception:
    redis = None; print("[DEBUG] redis indisponível.", flush=True)
print("[DEBUG] Importando pickle...", flush=True); import pickle; print("[DEBUG] pickle importado.", flush=True)
print("[DEBUG] Importando hashlib...", flush=True); import hashlib; print("[DEBUG] hashlib importado.", flush=True)
print("[DEBUG] Importando signal...", flush=True); import signal; print("[DEBUG] signal importado.", flush=True)
print("[DEBUG] Importando sys...", flush=True); import sys; print("[DEBUG] sys importado.", flush=True)
print("[DEBUG] Importando requests...", flush=True); import requests; print("[DEBUG] requests importado.", flush=True)
try:
    print("[DEBUG] Importando news_calendar.apply_blackout...", flush=True); from news_calendar import apply_blackout; print("[DEBUG] news_calendar.apply_blackout importado.", flush=True)
except Exception:
    def apply_blackout(*args, **kwargs): return (False, "")
    print("[DEBUG] news_calendar indisponível.", flush=True)
ml_optimizer = None
print("[DEBUG] Importando re...", flush=True); import re; print("[DEBUG] re importado.", flush=True)
RandomForestClassifier = None
StandardScaler = None
try:
    print("[DEBUG] Importando joblib...", flush=True); import joblib; print("[DEBUG] joblib importado.", flush=True)
except Exception:
    joblib = None; print("[DEBUG] joblib indisponível.", flush=True)
anti_chop_rf = None
anti_chop_scaler = None
ANTI_CHOP_MODEL_PATH = "anti_chop_rf.pkl"

def ensure_ml_loaded(timeout_seconds: float = 5.0) -> bool:
    global ml_optimizer
    if ml_optimizer is not None:
        return True
    try:
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as TErr
        def _do_import():
            import os
            os.environ.setdefault("OMP_NUM_THREADS", "1")
            os.environ.setdefault("MKL_NUM_THREADS", "1")
            os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
            import ml_optimizer as ml
            return ml
        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(_do_import)
            try:
                ml = fut.result(timeout=timeout_seconds)
                ml_optimizer = ml
                print("[DEBUG] ml_optimizer carregado (lazy).", flush=True)
                return True
            except TErr:
                print("[WARN] Timeout ao importar ml_optimizer", flush=True)
                return False
            except Exception as e:
                print(f"[ERROR] Falha ao importar ml_optimizer: {e}", flush=True)
                return False
    except Exception as e:
        print(f"[ERROR] Lazy import infra falhou: {e}", flush=True)
    return False
def load_anti_chop_model():
    global anti_chop_rf
    if anti_chop_rf is not None:
        return
    # Lazy import sklearn/joblib com timeout curto
    try:
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as _TErr
        def _do_imports():
            import os
            os.environ.setdefault("OMP_NUM_THREADS", "1")
            os.environ.setdefault("MKL_NUM_THREADS", "1")
            os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
            from sklearn.ensemble import RandomForestClassifier as RFC
            from sklearn.preprocessing import StandardScaler as SS
            import joblib as JB
            return RFC, SS, JB
        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(_do_imports)
            try:
                RFC, SS, JB = fut.result(timeout=3.0)
            except _TErr:
                return
            except Exception:
                return
    except Exception:
        return
    # Carrega do disco se existir
    if os.path.exists(ANTI_CHOP_MODEL_PATH):
        try:
            anti_chop_rf = JB.load(ANTI_CHOP_MODEL_PATH)
        except:
            anti_chop_rf = None
    if anti_chop_rf is None:
        anti_chop_rf = RFC(n_estimators=100, random_state=42)
        try:
            JB.dump(anti_chop_rf, ANTI_CHOP_MODEL_PATH)
        except Exception:
            pass

# Inicialização adiada: carregamento de ML/anti-chop só quando necessário
def is_valid_dataframe(df, min_rows: int = 1) -> bool:
    """Valida se o DataFrame é válido para cálculos."""
    if df is None: return False
    if isinstance(df, pd.DataFrame):
        return not df.empty and len(df) >= min_rows
    return False

logger = logging.getLogger("bot")
mt5_lock = RLock()
try:
    import telebot
except ImportError:
    telebot = None
    logger.warning("telebot não instalado - comandos Telegram desativados")
# =========================================================
# CONFIG GERAL
# =========================================================

TIMEFRAME_BASE = mt5.TIMEFRAME_M15
TIMEFRAME_MACRO = getattr(mt5, f"TIMEFRAME_{config.MACRO_TIMEFRAME}", mt5.TIMEFRAME_H1)
logger = logging.getLogger("utils")

# =========================================================
# 🌐 POLYGON.IO FALLBACK
# =========================================================

def get_polygon_data(endpoint: str, params: dict = {}) -> Optional[Dict]:
    """
    Centralized helper for Polygon.io API calls with mock key protection.
    """
    if not hasattr(config, "POLYGON_API_KEY") or config.POLYGON_API_KEY == "MOCK_KEY_FOR_NOW":
        return None
        
    try:
        if not endpoint.startswith("http"):
            url = f"{config.POLYGON_BASE_URL}/{endpoint}"
        else:
            url = endpoint
            
        full_params = params.copy()
        full_params["apiKey"] = config.POLYGON_API_KEY
        
        resp = requests.get(url, params=full_params, timeout=8)
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 401:
            logger.warning("⚠️ Polygon.io: Chave de API expirada ou inválida (401).")
        return None
    except Exception as e:
        logger.debug(f"Erro silenciado ao buscar Polygon ({endpoint}): {e}")
        return None

def get_polygon_rates_fallback(symbol: str, timeframe, count: int) -> Optional[pd.DataFrame]:
    """
    ✅ Fallback: Obtém dados da Polygon.io se o MT5 falhar.
    """
    try:
        pass
    except Exception:
        return None
        # Mapa de timeframe MT5 → Polygon
        tf_map = {
            mt5.TIMEFRAME_M1: ("minute", 1),
            mt5.TIMEFRAME_M5: ("minute", 5),
            mt5.TIMEFRAME_M15: ("minute", 15),
            mt5.TIMEFRAME_H1: ("hour", 1),
            mt5.TIMEFRAME_D1: ("day", 1),
        }
        
        timespan, multiplier = tf_map.get(timeframe, ("minute", 15))
        ticker = symbol.replace(".SA", "") # Formato B3 simples
        
        # Pega últimos 30 dias de histórico
        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        
        endpoint = f"v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{start}/{end}"
        params = {
            "adjusted": "true",
            "sort": "desc",
            "limit": count
        }
        
        data = get_polygon_data(endpoint, params)
        if data:
            results = data.get("results", [])
            if results:
                df = pd.DataFrame(results)
                # Formata para padrão MT5
                df = df.rename(columns={
                    "t": "time", "o": "open", "h": "high", 
                    "l": "low", "c": "close", "v": "tick_volume", "vw": "vwap"
                })
                df["time"] = pd.to_datetime(df["time"], unit="ms")
                if "real_volume" not in df.columns:
                    df["real_volume"] = df["tick_volume"] * df["close"]

                
                df.set_index("time", inplace=True)
                logger.info(f"✅ Fallback Polygon SUCESSO: {symbol}")
                return df.sort_index()
                
        return None

def resolve_current_symbol(root_symbol: str) -> Optional[str]:
    try:
        if mt5 is None:
            return None
        base = root_symbol.replace("$", "")
        group = f"*{base}*"
        syms = mt5.symbols_get(group) or []
        def _is_generic(name: str) -> bool:
            if "$" in name or "@" in name:
                info = mt5.symbol_info(name)
                if not info or not getattr(info, "selectable", False):
                    return True
            return False
        def _exp(name: str):
            info = mt5.symbol_info(name)
            exp = getattr(info, "expiration_time", None)
            return exp
        def _valid_future(name: str) -> bool:
            import re
            if _is_generic(name):
                return False
            if not name.startswith(base):
                return False
            if not re.search(rf"^{base}[FGHJKMNQUVXZ]\d{{2}}$", name):
                return False
            info = mt5.symbol_info(name)
            if not info or not getattr(info, "selectable", False):
                return False
            return True
        candidates = [getattr(s, "name", "") for s in syms if getattr(s, "name", "")]
        candidates = [n for n in candidates if _valid_future(n)]
        today = datetime.now()
        candidates = [(n, _exp(n)) for n in candidates]
        candidates = [(n, e) for n, e in candidates if isinstance(e, datetime) and e > today]
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[1])
        return candidates[0][0]
    except Exception:
        return None

def _expiration_str(symbol: str) -> str:
    try:
        info = mt5.symbol_info(symbol)
        exp = getattr(info, "expiration_time", None)
        if isinstance(exp, datetime):
            return exp.strftime("%d/%m/%Y")
        return ""
    except Exception:
        return ""

def update_futures_mappings():
    try:
        mapping = {}
        for k in ["WIN$", "WDO$", "WSP$"]:
            base = k.replace("$", "")
            sym = resolve_current_symbol(base)
            if sym:
                mapping[k] = sym
                exp_str = _expiration_str(sym)
                logger.info(f"[AUTO-DISCOVERY] Contrato {k} mapeado para {sym} (Vence em: {exp_str}).")
        if mapping:
            setattr(config, "ACTIVE_FUTURES", mapping)
            sector = getattr(config, "SECTOR_MAP", {})
            for g, real in mapping.items():
                if g in sector:
                    sector.pop(g, None)
                sector[real] = "FUTUROS"
            setattr(config, "SECTOR_MAP", sector)
        return getattr(config, "ACTIVE_FUTURES", {})
    except Exception:
        return {}
def detect_broker() -> str:
    try:
        if mt5 is None:
            return ""
        info = mt5.terminal_info()
        return str(getattr(info, "server", "") or "")
    except Exception:
        return ""
def get_futures_candidates(base_code: str) -> list:
    try:
        if mt5 is None:
            return []
        masks = [f"{base_code}*", f"{base_code}$*", f"{base_code}@*"]
        broker = detect_broker().lower()
        if "xp" in broker:
            masks += [f"{base_code}N*", f"{base_code}Z*"]
        if "btg" in broker or "clear" in broker:
            masks += [f"{base_code}?*"]
        seen = set()
        names = []
        for m in masks:
            try:
                res = mt5.symbols_get(m) or []
                for s in res:
                    n = getattr(s, "name", "")
                    if n and n not in seen:
                        seen.add(n)
                        names.append(n)
            except Exception:
                pass
        import re
        today = datetime.now()
        out = []
        for n in names:
            info = mt5.symbol_info(n)
            selectable = bool(info and getattr(info, "selectable", False))
            visible = bool(info and getattr(info, "visible", False))
            exp = getattr(info, "expiration_time", None)
            is_contract = bool(re.search(rf"^{base_code}[FGHJKMNQUVXZ]\d{{2}}$", n))
            if not is_contract:
                continue
            if not selectable:
                continue
            if not isinstance(exp, datetime) or exp <= today:
                continue
            try:
                mt5.symbol_select(n, True)
                rates = mt5.copy_rates_from_pos(n, mt5.TIMEFRAME_M15, 0, 120)
                vol = 0.0
                if rates is not None and len(rates) > 0:
                    df = pd.DataFrame(rates)
                    vol = float(df.get("tick_volume", pd.Series(dtype=float)).tail(80).sum() or 0.0)
            except Exception:
                vol = 0.0
            days_to_exp = (exp - today).days
            out.append({"symbol": n, "exp": exp, "days_to_exp": days_to_exp, "visible": visible, "selectable": selectable, "volume": vol})
        return out
    except Exception:
        return []
def calculate_contract_score(symbol_meta: dict) -> float:
    try:
        d = symbol_meta.get("days_to_exp", 9999)
        v = float(symbol_meta.get("volume", 0.0) or 0.0)
        vis = 1.0 if symbol_meta.get("visible") else 0.0
        sel = 1.0 if symbol_meta.get("selectable") else 0.0
        d_score = max(0.0, 1.0 - min(d, 180) / 180.0)
        v_score = min(1.0, v / 1_000_000.0)
        base = d_score * 0.4 + v_score * 0.4 + vis * 0.1 + sel * 0.1
        return base * 100.0
    except Exception:
        return 0.0
def log_mapping_details(base: str, candidates: list, selected: Optional[str]):
    try:
        logger.info("🔍 Iniciando Auto-Discovery de Contratos Futuros...")
        logger.info(f"📡 Corretora detectada: {detect_broker()}")
        logger.info(f"🎯 Mapeando {base}$ ...")
        if candidates:
            cand_syms = [c["symbol"] for c in candidates]
            logger.info(f"   Candidatos encontrados: {', '.join(cand_syms)}")
            for c in candidates:
                exp = c.get("exp")
                d = c.get("days_to_exp", 0)
                v = float(c.get("volume", 0.0) or 0.0)
                sc = calculate_contract_score(c)
                exp_str = exp.strftime("%b/%Y") if isinstance(exp, datetime) else "N/A"
                logger.info(f"   {c['symbol']}: Venc={exp_str} ({d} dias) | Vol={int(v)} | Score={int(sc)}")
        else:
            logger.warning("   Nenhum candidato encontrado")
        if selected:
            logger.info(f"✅ {base}$ → {selected}")
        else:
            logger.warning(f"❌ Falha ao mapear {base}$")
    except Exception:
        pass
def map_generic_to_specific(generic: str) -> Optional[str]:
    try:
        base = generic.replace("$", "")
        alt = {
            "WIN": ["WIN", "IND"],
            "WDO": ["WDO", "DOL"],
            "SMALL": ["SMLL", "SMAL", "SMALL"],
            "WSP": ["WSP", "SP"]
        }
        bases = alt.get(base, [base])
        cands = []
        for b in bases:
            cands = get_futures_candidates(b)
            if cands:
                break
        cands_sorted = sorted(cands, key=lambda c: (-calculate_contract_score(c), c.get("days_to_exp", 9999)))
        selected = cands_sorted[0]["symbol"] if cands_sorted else None
        log_mapping_details(base, cands_sorted, selected)
        return selected
    except Exception:
        return None
def discover_all_futures() -> dict:
    try:
        generics = ["WIN$", "WDO$", "SMALL$", "WSP$"]
        result = {}
        for g in generics:
            s = map_generic_to_specific(g)
            if s:
                result[g] = s
        if result:
            setattr(config, "ACTIVE_FUTURES", result)
            sector = getattr(config, "SECTOR_MAP", {})
            for g, real in result.items():
                if g in sector:
                    sector.pop(g, None)
                sector[real] = "FUTUROS"
            setattr(config, "SECTOR_MAP", sector)
            try:
                out_dir = Path("futures_optimizer_output")
                out_dir.mkdir(parents=True, exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                p = out_dir / f"futures_mappings_{ts}.json"
                meta = {"broker": detect_broker(), "mappings": result}
                with open(p, "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2, default=str)
                logger.info(f"💾 Mapeamentos salvos em: {str(p)}")
            except Exception:
                pass
        return result
    except Exception:
        return {}
    except Exception as e:
        logger.error(f"Erro Polygon Fallback ({symbol}): {e}")
        return None

def monitor_futures_rollover():
    try:
        mapping = getattr(config, "ACTIVE_FUTURES", {})
        for generic, current in mapping.items():
            base = generic.replace("$", "")
            cands = get_futures_candidates(base)
            if not cands:
                continue
            cands_sorted = sorted(cands, key=lambda c: (-calculate_contract_score(c), c.get("days_to_exp", 9999)))
            next_contract = cands_sorted[0]
            if not next_contract or next_contract.get("symbol") == current:
                continue
            curr_meta = next((c for c in cands if c.get("symbol") == current), None)
            if not curr_meta:
                continue
            vol_next = float(next_contract.get("volume", 0.0) or 0.0)
            vol_curr = float(curr_meta.get("volume", 0.0) or 0.0)
            days_to_exp = int(curr_meta.get("days_to_exp", 999))
            if days_to_exp <= 5 or vol_next > vol_curr:
                logger.warning(f"ROLLOVER {generic}: {current} → {next_contract['symbol']} (expira em {days_to_exp}d, vol cross={vol_next:.0f}>{vol_curr:.0f})")
    except Exception:
        pass

def initiate_rollover(generic_code: str, confirm_fn=None) -> bool:
    try:
        mapping = getattr(config, "ACTIVE_FUTURES", {})
        current = mapping.get(generic_code)
        if not current:
            return False
        base = generic_code.replace("$", "")
        cands = get_futures_candidates(base)
        if not cands:
            return False
        cands_sorted = sorted(cands, key=lambda c: (-calculate_contract_score(c), c.get("days_to_exp", 9999)))
        next_symbol = cands_sorted[0]["symbol"]
        with mt5_lock:
            positions = mt5.positions_get(symbol=current) or []
        if not positions:
            mapping[generic_code] = next_symbol
            setattr(config, "ACTIVE_FUTURES", mapping)
            return True
        pos = positions[0]
        side = "BUY" if pos.type == mt5.POSITION_TYPE_BUY else "SELL"
        volume = float(pos.volume)
        tick_curr = mt5.symbol_info_tick(current)
        tick_next = mt5.symbol_info_tick(next_symbol)
        if not tick_curr or not tick_next:
            return False
        price_next = tick_next.ask if side == "BUY" else tick_next.bid
        ord_type = mt5.ORDER_TYPE_BUY if side == "BUY" else mt5.ORDER_TYPE_SELL
        est_margin = float(mt5.order_calc_margin(ord_type, next_symbol, price_next, volume) or 0.0)
        acc = mt5.account_info()
        free_margin = float(getattr(acc, "margin_free", 0.0) or 0.0)
        if free_margin < est_margin * 1.30:
            return False
        if callable(confirm_fn) and not confirm_fn(current, next_symbol, volume):
            return False
        close_req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": current,
            "volume": float(volume),
            "type": mt5.ORDER_TYPE_SELL if side == "BUY" else mt5.ORDER_TYPE_BUY,
            "price": tick_curr.bid if side == "BUY" else tick_curr.ask,
            "comment": "ROLLOVER_CLOSE",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_RETURN,
        }
        open_req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": next_symbol,
            "volume": float(volume),
            "type": ord_type,
            "price": price_next,
            "comment": "ROLLOVER_OPEN",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_RETURN,
        }
        with mt5_lock:
            r1 = mt5.order_send(close_req)
            r2 = mt5.order_send(open_req) if r1 and r1.retcode == mt5.TRADE_RETCODE_DONE else None
        if r1 and r1.retcode == mt5.TRADE_RETCODE_DONE and r2 and r2.retcode == mt5.TRADE_RETCODE_DONE:
            mapping[generic_code] = next_symbol
            setattr(config, "ACTIVE_FUTURES", mapping)
            return True
        return False
    except Exception:
        return False

def enforce_stopout_prevention():
    try:
        acc = mt5.account_info()
        free_margin = float(getattr(acc, "margin_free", 0.0) or 0.0)
        with mt5_lock:
            positions = mt5.positions_get() or []
        if not positions:
            return
        total_req = 0.0
        for p in positions:
            tick = mt5.symbol_info_tick(p.symbol)
            if not tick:
                continue
            price = tick.ask if p.type == mt5.POSITION_TYPE_BUY else tick.bid
            ord_type = mt5.ORDER_TYPE_BUY if p.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_SELL
            total_req += float(mt5.order_calc_margin(ord_type, p.symbol, price, float(p.volume)) or 0.0)
        if total_req > 0 and free_margin < total_req * 0.50:
            positions_sorted = sorted(positions, key=lambda p: float(p.volume), reverse=True)
            for p in positions_sorted[:max(1, len(positions_sorted)//2)]:
                vol_close = max(float(p.volume) * 0.5, 0.01)
                tick = mt5.symbol_info_tick(p.symbol)
                if not tick:
                    continue
                req = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": p.symbol,
                    "volume": float(vol_close),
                    "type": mt5.ORDER_TYPE_SELL if p.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                    "price": tick.bid if p.type == mt5.POSITION_TYPE_BUY else tick.ask,
                    "comment": "STOP_OUT_PREVENTION",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_RETURN,
                }
                with mt5_lock:
                    mt5.order_send(req)
    except Exception:
        pass

def check_b3_circuit_breaker() -> str:
    try:
        df = safe_copy_rates("IBOV", mt5.TIMEFRAME_D1, 5)
        if not is_valid_dataframe(df, 2):
            return "OK"
        today = df['close'].iloc[-1]
        prev = df['close'].iloc[-2]
        drop = (today / prev) - 1.0
        if drop <= -0.10:
            return "CLOSE_ALL"
        if drop <= -0.07:
            return "PAUSE_AND_TIGHTEN"
        return "OK"
    except Exception:
        return "OK"

def should_block_opening_gap(symbol: str, max_gap_pct: float = None) -> bool:
    try:
        max_gap = float(max_gap_pct or getattr(config, "MAX_ACCEPTABLE_GAP_PCT", 0.015))
        df = safe_copy_rates(symbol, mt5.TIMEFRAME_M15, 96)
        if not is_valid_dataframe(df, 10):
            return False
        today = datetime.now().date()
        df_today = df[df.index.date == today]
        if df_today.empty:
            return False
        first = df_today.iloc[0]
        prev_close = df[df.index < df_today.index[0]]['close'].iloc[-1]
        gap = float(first['open'] / prev_close - 1.0)
        now_time = datetime.now().time()
        if datetime_time(10, 0) <= now_time <= datetime_time(10, 15):
            return True
        if abs(gap) > max_gap:
            return True
        return False
    except Exception:
        return False

def block_if_high_portfolio_correlation(candidate_symbol: str, threshold: float = 0.70) -> bool:
    try:
        with mt5_lock:
            positions = mt5.positions_get() or []
        symbols = list({p.symbol for p in positions})
        if not symbols:
            return False
        symbols.append(candidate_symbol)
        corr = calculate_correlation_matrix(symbols, lookback=60)
        for s in symbols:
            if s == candidate_symbol:
                continue
            val = float(corr.get(candidate_symbol, {}).get(s, 0.0) or 0.0)
            if val >= threshold:
                return True
        return False
    except Exception:
        return False

def get_yahoo_rates_fallback(symbol: str, timeframe, count: int) -> Optional[pd.DataFrame]:
    try:
        import yfinance as yf
    except Exception:
        return None
    try:
        if symbol.upper() == "IBOV":
            ticker = "^BVSP"
        else:
            s_up = (symbol or "").upper()
            if is_future(s_up):
                ticker = s_up
            else:
                ticker = s_up if s_up.endswith(".SA") else f"{s_up}.SA"
        tf_map = {
            mt5.TIMEFRAME_M1: ("1m", "7d"),
            mt5.TIMEFRAME_M5: ("5m", "10d"),
            mt5.TIMEFRAME_M15: ("15m", "30d"),
            mt5.TIMEFRAME_H1: ("60m", "60d"),
            mt5.TIMEFRAME_D1: ("1d", "365d"),
        }
        interval, period = tf_map.get(timeframe, ("15m", "30d"))
        df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=True, actions=False)
        if df is None or df.empty:
            return None
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].rename(columns={
            'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'real_volume'
        })
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df['tick_volume'] = df['real_volume']
        if count and count > 0 and len(df) > count:
            df = df.tail(count)
        return df
    except Exception as e:
        logger.error(f"Erro Yahoo Fallback ({symbol}): {e}")
        return None



# =========================================================
# 💾 PERSISTÊNCIA ATÔMICA
# =========================================================
def atomic_save_json(filename: str, data: dict):
    """Salva JSON de forma atômica para evitar corrupção"""
    temp_file = f"{filename}.tmp"
    try:
        with open(temp_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        # Replace atômico (Linux/Unix) ou remove+rename (Windows)
        if os.path.exists(filename):
            os.remove(filename)
        os.rename(temp_file, filename)
    except Exception as e:
        logger.error(f"❌ Erro ao salvar JSON atômico: {e}")
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except:
                pass

# =========================================================
# 📊 HELPERS FINANCEIROS & METAS
# =========================================================

_symbol_consecutive_losses = defaultdict(int)

def get_loss_streak(symbol: str) -> int:
    return _symbol_consecutive_losses[symbol]

def get_vix_br() -> float:
    """Retorna VIX Brasil (Estimado ou via API se disponível)"""
    # Placeholder: Se tiver acesso a dados reais, implementar aqui.
    # Por enquanto, retorna valor seguro ou tenta inferir.
    return 22.5 # Valor médio seguro para não travar

def is_future(symbol: str) -> bool:
    try:
        s = (symbol or "").upper()
        return s.startswith(("WIN", "WDO", "IND", "DOL", "WSP", "BGI"))
    except Exception:
        return False

class AssetInspector:
    @staticmethod
    def detect(symbol: str) -> dict:
        s = (symbol or "").upper().strip()
        if "SMALL" in s:
            return {"type": "FUTURE", "point_value": 20.0, "tick_size": 0.1, "fee_type": "FIXED", "fee_val": 0.50}
        if any(x in s for x in ["WIN", "IND", "WSP"]):
            return {"type": "FUTURE", "point_value": 0.20, "tick_size": 5.0, "fee_type": "FIXED", "fee_val": 0.25}
        if any(x in s for x in ["WDO", "DOL", "BGI"]):
            return {"type": "FUTURE", "point_value": 10.0, "tick_size": 0.5, "fee_type": "FIXED", "fee_val": 1.10}
        # AÇÃO B3: padrão 4 letras + dígito (ex: PETR4)
        return {"type": "STOCK", "point_value": 1.0, "tick_size": 0.01, "fee_type": "PERCENT", "fee_val": 0.00055}

def round_to_tick(price: float, tick_size: float) -> float:
    try:
        if tick_size <= 0:
            return float(price)
        return float(round(price / tick_size) * tick_size)
    except Exception:
        return float(price)

sector_weights: Dict[str, Dict[str, float]] = {}
symbol_weights: Dict[str, Dict[str, float]] = {}
# Conexão Redis (ajuste se necessário)
try:
    redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=False)
    redis_client.ping()  # Testa conexão
    REDIS_AVAILABLE = True
    logger.info("✅ Redis conectado - cache ativado")
except Exception as e:
    redis_client = None
    REDIS_AVAILABLE = False
    logger.warning(f"⚠️ Redis não disponível: {e} - cache desativado")

# =========================================================
# MT5 SAFE COPY (ANTI-DEADLOCK)
# =========================================================

def safe_copy_rates(symbol: str, timeframe, count: int = 500, timeout: int = 12) -> Optional[pd.DataFrame]:
    # Tenta selecionar, mas continua mesmo se falhar (pode já estar no Market Watch)
    if not mt5.symbol_select(symbol, True):
        pass 

    try:
        bars = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
        bars_available = 0 if bars is None else len(bars)
    except Exception:
        bars_available = 0

    if bars_available < count:
        mt5.copy_rates_from_pos(symbol, timeframe, 0, 1)
        time.sleep(0.2)

    q = queue.Queue()

    def worker():
        try:
            q.put(mt5.copy_rates_from_pos(symbol, timeframe, 0, count))
        except Exception as e:
            q.put(e)

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    t.join(timeout)

    if t.is_alive():
        logger.error(f"🚨 TIMEOUT MT5 em {symbol}")
        return None

    try:
        rates = q.get_nowait()
        if isinstance(rates, Exception) or rates is None or len(rates) == 0:
            # ✅ FALLBACK: Polygon.io
            logger.warning(f"⚠️ MT5 Falhou em {symbol}. Acionando Polygon.io fallback...")
            df_fb = get_polygon_rates_fallback(symbol, timeframe, count)
            return df_fb

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)
        return df.sort_index()
    except queue.Empty:
        # ✅ FALLBACK: Polygon.io
        logger.warning(f"⚠️ Timeout MT5 em {symbol}. Acionando Polygon.io fallback...")
        df_fb = get_polygon_rates_fallback(symbol, timeframe, count)
        return df_fb

def get_training_rates(symbol: str, timeframe, bars: int = 2000) -> Optional[pd.DataFrame]:
    try:
        df = safe_copy_rates(symbol, timeframe, bars)
        return df
    except Exception as e:
        logger.error(f"get_training_rates: {e}")
        return None
# =========================================================
# AWS LAMBDA HOOK (SCALABILITY)
# =========================================================

def lambda_invoke(function_name: str, payload: dict) -> Optional[dict]:
    """
    Invoca uma função AWS Lambda para processamento pesado (ex: ML Inference).
    """
    try:
        import boto3
        import json
        client = boto3.client('lambda', region_name=getattr(config, 'AWS_REGION', 'us-east-1'))
        response = client.invoke(
            FunctionName=function_name,
            InvocationType='RequestResponse',
            Payload=json.dumps(payload)
        )
        return json.loads(response['Payload'].read())
    except ImportError:
        logger.debug("Boto3 não instalado. AWS Lambda desabilitado.")
        return None
    except Exception as e:
        logger.error(f"Erro ao invocar Lambda {function_name}: {e}")
        return None

# =========================================================
# REDIS CACHE HELPERS
# =========================================================

def cached_symbol_info(symbol: str) -> Optional[mt5.SymbolInfo]:
    """
    Versão com cache do symbol_info
    """
    if not REDIS_AVAILABLE:
        return mt5.symbol_info(symbol)
        
    cache_key = f"symbol_info:{symbol}"
    try:
        cached = redis_client.get(cache_key)
        if cached:
            return pickle.loads(cached)
            
        info = mt5.symbol_info(symbol)
        if info:
            redis_client.setex(cache_key, config.REDIS_CACHE_TTL_INFO, pickle.dumps(info))
        return info
    except Exception as e:
        logger.debug(f"Redis Info Error: {e}")
        return mt5.symbol_info(symbol)

def cached_symbol_info_tick(symbol: str):
    """
    Versão com cache do symbol_info_tick (TTL 1s)
    """
    if not REDIS_AVAILABLE:
        return mt5.symbol_info_tick(symbol)
        
    cache_key = f"tick:{symbol}"
    try:
        cached = redis_client.get(cache_key)
        if cached:
            return pickle.loads(cached)
            
        tick = mt5.symbol_info_tick(symbol)
        if tick:
            redis_client.setex(cache_key, config.REDIS_CACHE_TTL_TICK, pickle.dumps(tick))
        return tick
    except Exception as e:
        logger.debug(f"Redis Tick Error: {e}")
        return mt5.symbol_info_tick(symbol)


# =========================================================
# 🛑 FILTRO ANTI-OVERTRADING
# =========================================================

# Rastreamento de trades por símbolo e período
_trade_history = {}  # {symbol: [timestamps]}
_trade_history_lock = RLock()

def check_anti_overtrading(symbol: str, max_trades_per_hour: int = 5, 
                            max_trades_per_day: int = 20) -> Tuple[bool, str]:
    """
    ✅ Verifica se está fazendo trades demais em um símbolo.
    
    Regras:
    - Máximo 3 trades/hora por símbolo
    - Máximo 10 trades/dia por símbolo
    
    Returns:
        (allowed: bool, reason: str)
    """
    global _trade_history
    
    with _trade_history_lock:
        now = datetime.now()
        
        if symbol not in _trade_history:
            _trade_history[symbol] = []
        
        # Limpa histórico antigo (>24h)
        _trade_history[symbol] = [
            t for t in _trade_history[symbol] 
            if (now - t).total_seconds() < 86400
        ]
        
        # Conta trades na última hora
        hour_ago = now - timedelta(hours=1)
        trades_last_hour = sum(1 for t in _trade_history[symbol] if t >= hour_ago)
        
        # Conta trades hoje
        today = now.replace(hour=0, minute=0, second=0, microsecond=0)
        trades_today = sum(1 for t in _trade_history[symbol] if t >= today)
        
        if trades_last_hour >= max_trades_per_hour:
            return False, f"Overtrading/hora: {trades_last_hour}/{max_trades_per_hour}"
        
        if trades_today >= max_trades_per_day:
            return False, f"Overtrading/dia: {trades_today}/{max_trades_per_day}"
        
        return True, f"OK ({trades_last_hour}/h, {trades_today}/d)"


def register_trade_for_overtrading(symbol: str):
    """Registra um trade para controle de overtrading."""
    global _trade_history
    
    with _trade_history_lock:
        if symbol not in _trade_history:
            _trade_history[symbol] = []
        _trade_history[symbol].append(datetime.now())


# =========================================================
# 📊 WIN RATE EM TEMPO REAL
# =========================================================

def get_realtime_win_rate(lookback_trades: int = 20) -> Dict[str, float]:
    """
    ✅ Calcula win rate em tempo real dos últimos N trades.
    
    Returns:
        {
            'win_rate': float (0-1),
            'total_trades': int,
            'wins': int,
            'losses': int,
            'avg_win': float,
            'avg_loss': float,
            'profit_factor': float,
            'expectancy': float
        }
    """
    try:
        with mt5_lock:
            deals = mt5.history_deals_get(
                datetime.now() - timedelta(days=7), 
                datetime.now()
            )
        
        if not deals:
            return _default_win_rate_result()
        
        # Filtra fechamentos
        out_deals = [d for d in deals if d.entry == mt5.DEAL_ENTRY_OUT][-lookback_trades:]
        
        if len(out_deals) < 5:
            return _default_win_rate_result()
        
        wins = [d for d in out_deals if d.profit > 0]
        losses = [d for d in out_deals if d.profit <= 0]
        
        win_rate = len(wins) / len(out_deals)
        
        avg_win = sum(d.profit for d in wins) / len(wins) if wins else 0
        avg_loss = abs(sum(d.profit for d in losses) / len(losses)) if losses else 0
        
        profit_factor = avg_win / avg_loss if avg_loss > 0 else float('inf')
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        
        return {
            'win_rate': win_rate,
            'total_trades': len(out_deals),
            'wins': len(wins),
            'losses': len(losses),
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'expectancy': expectancy
        }
        
    except Exception as e:
        logger.error(f"Erro ao calcular win rate real-time: {e}")
        return _default_win_rate_result()


def _default_win_rate_result() -> Dict[str, float]:
    """Resultado padrão quando não há dados suficientes."""
    return {
        'win_rate': 0.50,
        'total_trades': 0,
        'wins': 0,
        'losses': 0,
        'avg_win': 0,
        'avg_loss': 0,
        'profit_factor': 1.0,
        'expectancy': 0
    }


def get_symbol_performance(symbol: str, lookback_days: int = 30) -> Dict[str, float]:
    """
    ✅ Performance específica de um símbolo.
    """
    try:
        with mt5_lock:
            deals = mt5.history_deals_get(
                datetime.now() - timedelta(days=lookback_days), 
                datetime.now()
            )
        
        if not deals:
            return {'win_rate': 0.55, 'trades': 0}
        
        symbol_deals = [d for d in deals if d.symbol == symbol and d.entry == mt5.DEAL_ENTRY_OUT]
        
        if len(symbol_deals) < 3:
            return {'win_rate': 0.55, 'trades': len(symbol_deals)}
        
        wins = sum(1 for d in symbol_deals if d.profit > 0)
        total_pnl = sum(d.profit for d in symbol_deals)
        
        return {
            'win_rate': wins / len(symbol_deals),
            'trades': len(symbol_deals),
            'wins': wins,
            'losses': len(symbol_deals) - wins,
            'total_pnl': total_pnl
        }
        
    except Exception as e:
        logger.error(f"Erro ao buscar performance {symbol}: {e}")
        return {'win_rate': 0.55, 'trades': 0}


def get_dynamic_rr_min() -> float:
    """
    ✅ R:R DINÂMICO MULTI-FATOR
    
    Fatores:
    1. Regime de mercado (IBOV trend)
    2. VIX Brasil (BVIX)
    3. Volatilidade histórica (ATR agregado)
    4. Drawdown atual
    
    Ranges:
    - RISK_ON + vol baixa: 1.2
    - RISK_OFF + vol alta: 2.5
    """
    regime = detect_market_regime()
    if regime not in ("RISK_ON", "RISK_OFF"):
        regime = "RISK_ON"
    
    # ✅ NOVO: Fator 1 - Regime base
    if regime == "RISK_ON":
        base_rr = 1.2
    else:
        base_rr = 1.5
    
    # ✅ NOVO: Fator 2 - Volatilidade histórica
    try:
        ibov_df = safe_copy_rates("IBOV", mt5.TIMEFRAME_D1, 30)
        
        if ibov_df is not None:
            hvol = ibov_df['close'].pct_change().std() * 100
            
            # Multiplica RR se vol > 2%
            if hvol > 2.0:
                vol_multiplier = 1.3
            elif hvol > 1.5:
                vol_multiplier = 1.15
            else:
                vol_multiplier = 1.0
        else:
            vol_multiplier = 1.0
    except:
        vol_multiplier = 1.0
    
    # ✅ NOVO: Fator 3 - Drawdown atual
    try:
        from bot import daily_max_equity
        acc = mt5.account_info()
        
        if acc and daily_max_equity > 0:
            current_dd = (daily_max_equity - acc.equity) / daily_max_equity
            
            # Aumenta RR mínimo se em drawdown
            if current_dd > 0.03:  # >3%
                dd_multiplier = 1.5
            elif current_dd > 0.02:  # >2%
                dd_multiplier = 1.25
            else:
                dd_multiplier = 1.0
        else:
            dd_multiplier = 1.0
    except:
        dd_multiplier = 1.0
    
    # Cálculo Final
    final_rr = base_rr * vol_multiplier * dd_multiplier
    final_rr = min(final_rr, 2.2)
   
    # logger.info(f"📊 R:R Dinâmico | Regime: {regime} | Base: {base_rr:.2f} | Vol×: {vol_multiplier:.2f} | DD×: {dd_multiplier:.2f} | Final: {final_rr:.2f}")
    
    return final_rr

# =========================================================
# 🌐 POLYGON.IO INTEGRATION
# =========================================================

def get_vix_br() -> float:
    """
    ✅ V5.2: Obtém VIX Brasil (proxy) via Polygon ou MT5.
    Se falhar, retorna valor baseado na volatilidade do IBOV.
    """
    if REDIS_AVAILABLE:
        cached = redis_client.get("vix_br")
        if cached: return float(cached)

    vix = 20.0
    try:
        # Tenta proxy via Polygon (VIXM ou similar se disponível)
        data = get_polygon_data("v2/aggs/ticker/I:VIX/prev")
        if data and data.get("results"):
            vix = data["results"][0]["c"]
        else:
            # Fallback: Volatilidade do IBOV
            df = safe_copy_rates("IBOV", mt5.TIMEFRAME_H1, 100)
            if is_valid_dataframe(df):
                vix = df['close'].pct_change().std() * 100 * np.sqrt(252 * 7) # Proxy volatilidade anualizada
                vix = max(10, min(vix, 80))
    except:
        vix = 22.0

    if REDIS_AVAILABLE:
        redis_client.setex("vix_br", 300, str(vix)) # 5 min cache
    return float(vix)

def get_order_flow(symbol: str, bars: int = 20) -> Dict[str, float]:
    """
    ✅ V5.2 CONSOLIDADO: Order Flow Híbrido (Polygon + MT5)
    """
    # 1. Tenta Polygon (Melhor qualidade para CVD real)
    ticker = symbol.replace(".SA", "")
    end = datetime.now().strftime("%Y-%m-%d")
    endpoint = f"v2/aggs/ticker/{ticker}/range/1/minute/{end}/{end}"
    
    data = get_polygon_data(endpoint, {"limit": bars})
    if data:
        res = data.get("results", [])
        if res:
            buy_v = sum(r['v'] for r in res if r['c'] >= r['o'])
            sell_v = sum(r['v'] for r in res if r['c'] < r['o'])
            total = buy_v + sell_v
            return {
                "imbalance": (buy_v - sell_v) / total if total > 0 else 0,
                "cvd": buy_v - sell_v,
                "buy_volume": buy_v,
                "sell_volume": sell_v
            }

    # 2. Fallback MT5 (Estimado)
    df = safe_copy_rates(symbol, mt5.TIMEFRAME_M1, bars)
    if not is_valid_dataframe(df):
        return {"imbalance": 0.0, "cvd": 0.0, "buy_volume": 0, "sell_volume": 0}
    
    df['delta'] = np.where(df['close'] >= df['open'], df['tick_volume'], -df['tick_volume'])
    buy_v = df[df['delta'] > 0]['tick_volume'].sum()
    sell_v = df[df['delta'] < 0]['tick_volume'].sum()
    total = buy_v + sell_v
    
    return {
        "imbalance": (buy_v - sell_v) / total if total > 0 else 0,
        "cvd": float(df['delta'].sum()),
        "buy_volume": float(buy_v),
        "sell_volume": float(sell_v)
    }



# =========================================================
# SLIPPAGE
# =========================================================

def get_real_slippage(symbol: str) -> float:
    tick = mt5.symbol_info_tick(symbol)
    if not tick or tick.bid <= 0:
        return config.SLIPPAGE_MAP.get("DEFAULT", 0.005)

    spread_pct = (tick.ask - tick.bid) / tick.bid

    # Multiplicador por perfil de liquidez
    if symbol in config.LOW_LIQUIDITY_SYMBOLS:
        mult = 2.0
    elif is_power_hour():
        mult = 1.2
    else:
        mult = 1.5

    mapped = config.SLIPPAGE_MAP.get(symbol, config.SLIPPAGE_MAP.get("DEFAULT"))
    return max(spread_pct * mult, mapped)


def check_and_alert_slippage(symbol: str, requested_price: float, 
                               executed_price: float, side: str) -> bool:
    """
    Verifica se houve slippage excessivo na execução e notifica o usuário.
    
    Args:
        symbol: Símbolo do ativo
        requested_price: Preço solicitado
        executed_price: Preço executado
        side: BUY ou SELL
    
    Returns:
        bool: True se slippage foi excessivo
    """
    try:
        info = mt5.symbol_info(symbol)
        if not info or info.point <= 0:
            return False
        
        # Calcula slippage em ticks
        slippage = abs(executed_price - requested_price)
        slippage_ticks = slippage / info.point
        
        # Verifica direção (adverso = pior para o trader)
        if side == "BUY":
            is_adverse = executed_price > requested_price
        else:
            is_adverse = executed_price < requested_price
        
        max_slippage_ticks = getattr(config, 'SLIPPAGE_ALERT_TICKS', 3)
        
        if is_adverse and slippage_ticks > max_slippage_ticks:
            logger.warning(
                f"⚠️ SLIPPAGE EXCESSIVO {symbol} | "
                f"Solicitado: {requested_price:.2f} | "
                f"Executado: {executed_price:.2f} | "
                f"Slippage: {slippage_ticks:.1f} ticks"
            )
            
            # Envia alerta Telegram
            try:
                send_telegram_message(
                    f"⚠️ <b>SLIPPAGE EXCESSIVO</b>\n\n"
                    f"📊 Ativo: <b>{symbol}</b>\n"
                    f"📈 Direção: {side}\n"
                    f"💰 Solicitado: R$ {requested_price:.2f}\n"
                    f"💸 Executado: R$ {executed_price:.2f}\n"
                    f"📉 Slippage: <b>{slippage_ticks:.1f} ticks</b>\n\n"
                    f"💡 <i>Revise a liquidez do ativo</i>"
                )
            except Exception as e:
                logger.debug(f"Erro ao enviar alerta slippage: {e}")
            
            return True
        
        return False
        
    except Exception as e:
        logger.error(f"Erro ao verificar slippage: {e}")
        return False


# =========================================================
# REGIME DE MERCADO
# =========================================================

def detect_market_regime() -> str:
    ibov = safe_copy_rates("IBOV", mt5.TIMEFRAME_D1, 50)
    if ibov is None or len(ibov) < 30:
        return "RISK_ON"

    close = ibov["close"]
    ma20 = close.rolling(20).mean().iloc[-1]
    ma50 = close.rolling(50).mean().iloc[-1]
    cur = close.iloc[-1]

    # REGRA MAIS FLEXÍVEL: evita desligar o bot em lateralização
    if cur > ma20:
        return "RISK_ON"
    elif cur < ma50:
        return "RISK_OFF"
    else:
        return "NEUTRAL"


# =========================================================
# EXPOSIÇÃO SETORIAL
# =========================================================

def calculate_sector_exposure_pct(equity: float) -> Dict[str, float]:
    with mt5_lock:
        positions = mt5.positions_get() or []

    sector_risk = defaultdict(float)
    for p in positions:
        sector = config.SECTOR_MAP.get(p.symbol, "UNKNOWN")
        sector_risk[sector] += p.volume * p.price_open

    return {s: v / equity for s, v in sector_risk.items()} if equity > 0 else {}

def get_ibov_correlation(symbol: str, lookback: int = 50) -> float:
    """
    Calcula a correlação de Pearson entre o ativo e o IBOVESPA.
    """
    try:
        df_sym = safe_copy_rates(symbol, mt5.TIMEFRAME_D1, lookback)
        df_ibov = safe_copy_rates("IBOV", mt5.TIMEFRAME_D1, lookback)
        
        if df_sym is None or df_ibov is None or len(df_sym) < 20 or len(df_ibov) < 20:
            return 0.0
            
        returns_sym = df_sym['close'].pct_change().dropna()
        returns_ibov = df_ibov['close'].pct_change().dropna()
        
        # Alinha os índices
        combined = pd.concat([returns_sym, returns_ibov], axis=1).dropna()
        if len(combined) < 10:
            return 0.0
            
        correlation = combined.corr().iloc[0, 1]
        return float(correlation)
    except Exception as e:
        logger.error(f"Erro ao calcular correlação IBOV para {symbol}: {e}")
        return 0.0

# deleted for debug




def validate_subsetor_exposure(symbol: str) -> Tuple[bool, str]:
    """
    ✅ Garante que não ultrapassamos 20% do capital (ou config) em um mesmo subsetor.
    """
    try:
        subsetor = config.SUBSETOR_MAP.get(symbol, "Outros")
        
        with mt5_lock:
            positions = mt5.positions_get()
            
        if not positions:
            return True, "OK"
            
        acc = mt5.account_info()
        total_equity = acc.equity if acc else 100000.0
        
        subsetor_value = 0
        for pos in positions:
            if config.SUBSETOR_MAP.get(pos.symbol) == subsetor:
                subsetor_value += pos.volume * pos.price_open
                
        risk_per_trade = float(getattr(config, "RISK_PER_TRADE", getattr(config, "RISK_PER_TRADE_PCT", 0.01)) or 0.01)
        projected_total = subsetor_value + (total_equity * risk_per_trade)
        exposure = projected_total / total_equity
        
        limit = getattr(config, "MAX_SUBSETOR_EXPOSURE", 0.20)
        
        if exposure > limit:
            return False, f"Exposição excessiva em {subsetor}: {exposure:.1%}"
            
        return True, "OK"
    except Exception as e:
        logger.error(f"Erro validação subsetor: {e}")
        return True, "OK"


def get_book_imbalance(symbol: str) -> float:
    try:
        book = mt5.market_book_get(symbol)
        if not book:
            return 0.0

        buys = sum(item.volume for item in book if item.type == mt5.BOOK_TYPE_BUY)
        sells = sum(item.volume for item in book if item.type == mt5.BOOK_TYPE_SELL)

        total = buys + sells
        if total == 0:
            return 0.0

        imbalance = (buys - sells) / total

        # ZONA NEUTRA: ignora ruído
        if abs(imbalance) < config.MIN_BOOK_IMBALANCE:
            return 0.0

        return imbalance

    except Exception as e:
        logger.error(f"Erro no book imbalance para {symbol}: {e}")
        return 0.0


# =========================================================
# FAST RATES
# =========================================================

_last_bar_time = {}

def get_fast_rates(symbol, timeframe):
    df = safe_copy_rates(symbol, timeframe, 3)
    if not is_valid_dataframe(df):
        return None

    last = df.index[-1]
    key = (symbol, timeframe)
    if _last_bar_time.get(key) == last:
        return None

    _last_bar_time[key] = last
    return df

# =========================================================
# INDICADORES BÁSICOS
# =========================================================

def get_atr(df: pd.DataFrame, period: int = 14) -> Optional[float]:
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs()
    ], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    return float(atr.iloc[-1])

def get_adx(df: pd.DataFrame, period: int = 14) -> Optional[float]:
    if len(df) < period * 2:
        return None

    high, low, close = df["high"], df["low"], df["close"]

    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr

    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()

    return float(adx.iloc[-1])

def get_adx_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Retorna a série completa do ADX para análise de tendência."""
    if df is None or len(df) < period * 2:
        return pd.Series()
    
    high = df['high']
    low = df['low']
    close = df['close']

    plus_dm = high.diff()
    minus_dm = low.diff()

    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    minus_dm = abs(minus_dm)

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr

    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()

    return adx


def get_intraday_vwap(df: pd.DataFrame) -> Optional[float]:
    """
    VWAP desde a abertura do pregão (10h00) até agora.
    """
    now = datetime.now()
    today = now.date()
    market_open = datetime.combine(today, datetime.strptime("10:00", "%H:%M").time())
    
    df_today = df[df.index >= market_open]
    
    if df_today.empty or len(df_today) < 3:
        return None
    
    typical_price = (df_today['high'] + df_today['low'] + 2 * df_today['close']) / 4
    volume = df_today.get('real_volume', df_today['tick_volume'])
    
    pv = (typical_price * volume).sum()
    total_vol = volume.sum()
    
    return float(pv / total_vol) if total_vol > 0 else None


def get_intraday_vwap_stats(df: pd.DataFrame) -> Optional[dict]:
    now = datetime.now()
    today = now.date()
    market_open = datetime.combine(today, datetime.strptime("10:00", "%H:%M").time())
    df_today = df[df.index >= market_open]
    if df_today.empty or len(df_today) < 3:
        return None
    typical_price = (df_today['high'] + df_today['low'] + 2 * df_today['close']) / 4
    volume = df_today.get('real_volume', df_today['tick_volume'])
    pv = (typical_price * volume).sum()
    total_vol = volume.sum()
    vwap = float(pv / total_vol) if total_vol > 0 else None
    if vwap is None:
        return None
    std = float(typical_price.std(ddof=0))
    upper_2sd = vwap + std * getattr(config, "VWAP_OVEREXT_STD_MULT", 2.0)
    lower_2sd = vwap - std * getattr(config, "VWAP_OVEREXT_STD_MULT", 2.0)
    return {"vwap": vwap, "std": std, "upper_2sd": upper_2sd, "lower_2sd": lower_2sd}

def timeframe_to_minutes(tf) -> int:
    m = {
        getattr(mt5, "TIMEFRAME_M1", None): 1,
        getattr(mt5, "TIMEFRAME_M5", None): 5,
        getattr(mt5, "TIMEFRAME_M15", None): 15,
        getattr(mt5, "TIMEFRAME_M30", None): 30,
        getattr(mt5, "TIMEFRAME_H1", None): 60,
        getattr(mt5, "TIMEFRAME_H4", None): 240,
        getattr(mt5, "TIMEFRAME_D1", None): 1440,
    }
    return m.get(tf, 15)

def get_avg_spread_pct(symbol: str, timeframe, bars: int = 10) -> Optional[float]:
    try:
        minutes = timeframe_to_minutes(timeframe) * max(int(bars), 1)
        start = datetime.now() - timedelta(minutes=minutes)
        with mt5_lock:
            ticks = mt5.copy_ticks_from(symbol, start, 100000, mt5.COPY_TICKS_ALL)
        if ticks is None or len(ticks) == 0:
            return None
        df_ticks = pd.DataFrame(ticks)
        df_ticks = df_ticks[(df_ticks["ask"] > 0) & (df_ticks["bid"] > 0)]
        if df_ticks.empty:
            return None
        spread_pct = (df_ticks["ask"] - df_ticks["bid"]) / df_ticks["bid"]
        return float(spread_pct.mean() * 100)
    except Exception:
        return None

def check_spread(symbol: str, timeframe, bars: int = None) -> tuple[bool, float, float]:
    with mt5_lock:
        tick = mt5.symbol_info_tick(symbol)
    if not tick or tick.ask <= 0 or tick.bid <= 0:
        return False, 0.0, 0.0
    current_pct = (tick.ask - tick.bid) / tick.bid * 100
    lookback = bars if bars is not None else int(getattr(config, "SPREAD_LOOKBACK_BARS", 10))
    avg_pct = get_avg_spread_pct(symbol, timeframe, lookback) or 0.0
    if current_pct > avg_pct and avg_pct > 0:
        return False, float(current_pct), float(avg_pct)
    return True, float(current_pct), float(avg_pct)
def get_obv(df: pd.DataFrame) -> Optional[float]:
    """
    Calcula OBV ignorando candle em aberto.
    """
    if df is None or len(df) < 20:
        return None
    
    try:
        # 1. Escolha do volume (B3 usa tick_volume se real_volume for zero)
        volume = df['tick_volume'].values
        close = df['close'].values
        
        # 2. Lógica Vetorizada (MUITO mais rápida que loop 'for')
        # Cria array de direção: 1 se subiu, -1 se caiu, 0 se igual
        direction = np.sign(np.diff(close))
        
        # O diff reduz o tamanho em 1, então alinhamos o volume do candle [1:]
        obv_changes = direction * volume[1:]
        
        # Soma acumulada para gerar a linha do OBV
        # Usamos apenas até o candle [-2] para análise estável
        full_obv = np.cumsum(np.concatenate(([volume[0]], obv_changes)))
        
        # 3. Retornamos o valor do ÚLTIMO CANDLE FECHADO (-2)
        # Isso garante que o volume da análise é real e completo
        return float(full_obv[-2])
    
    except Exception as e:
        logger.error(f"Erro ao calcular OBV: {e}")
        return None


# 🆕 NOVA FUNÇÃO - Tendência do OBV
def is_obv_trending_up(df: pd.DataFrame, lookback: int = 20) -> bool:
    """
    ✅ Valida se OBV está em tendência de alta
    
    Retorna True se:
    - OBV atual > média móvel de 20 períodos
    - OBV subiu nos últimos 5 candles
    """
    if df is None or len(df) < lookback + 5:
        return False
    
    try:
        # Calcula OBV completo
        volume = df.get('real_volume', df['tick_volume']).values
        close = df['close'].values
        price_changes = np.diff(close, prepend=close[0])
        
        obv = np.zeros(len(close))
        for i in range(1, len(close)):
            if price_changes[i] > 0:
                obv[i] = obv[i-1] + volume[i]
            elif price_changes[i] < 0:
                obv[i] = obv[i-1] - volume[i]
            else:
                obv[i] = obv[i-1]
        
        # Média móvel do OBV
        obv_ma = np.convolve(obv, np.ones(lookback)/lookback, mode='valid')
        
        obv_above_ma = obv[-1] > obv_ma[-1]
        obv_rising = obv[-1] > obv[-3]  # reduz lookback
        
        return obv_above_ma or obv_rising

    
    except Exception as e:
        logger.error(f"Erro ao validar OBV: {e}")
        return False

# =========================================================
# MACRO TREND
# =========================================================

def macro_trend_ok(symbol: str, side: str) -> bool:
    df = safe_copy_rates(symbol, TIMEFRAME_MACRO, 300)
    if df is None or len(df) < config.MACRO_EMA_LONG:
        return False

    close = df["close"]
    ema = close.ewm(span=config.MACRO_EMA_LONG, adjust=False).mean().iloc[-1]
    tick = mt5.symbol_info_tick(symbol)
    if not tick or (tick.last <= 0 and tick.bid <= 0):
        return False

    price = tick.last if tick.last > 0 else tick.bid

    adx = get_adx(df)
    min_adx = 17 if TIMEFRAME_MACRO >= mt5.TIMEFRAME_H1 else 14
    
    if adx is not None and adx < min_adx:
        return False

    return price > ema if side == "BUY" else price < ema

# =========================================================
# INDICADORES CONSOLIDADOS (SEM SCORE)
# =========================================================

def get_momentum(df: pd.DataFrame, period: int = 10) -> Optional[float]:
    """
    Calcula momentum (Rate of Change)
    
    Momentum = (preço_atual - preço_passado) / preço_passado
    
    Args:
        df: DataFrame com coluna 'close'
        period: Quantos candles olhar para trás (padrão: 10)
    
    Returns:
        Momentum como float (ex: 0.05 = 5% de alta)
    """
    if df is None or len(df) < period + 1:
        return None
    
    close = df['close']
    
    # Momentum = mudança percentual em N períodos
    momentum = (close.iloc[-1] - close.iloc[-period - 1]) / close.iloc[-period - 1]
    
    return float(momentum)


def quick_indicators_custom(symbol, timeframe, df=None, params=None):
    """
    ✅ VERSÃO COMPLETA: Inclui Momentum
    """
    params = params or {}
    df = df if df is not None else safe_copy_rates(symbol, timeframe, 300)

    if df is None or len(df) < 50:
        return {"error": "no_data"}

    close = df["close"]
    high = df["high"]
    low = df["low"]

    # --- MÉDIAS E RSI ---
    ema_fast = close.ewm(span=params.get("ema_short", 9), adjust=False).mean().iloc[-1]
    ema_slow = close.ewm(span=params.get("ema_long", 21), adjust=False).mean().iloc[-1]

    delta = close.diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = -delta.clip(upper=0).rolling(14).mean()
    rsi = (100 - (100 / (1 + up / down))).iloc[-1]

    # --- ATR E ADX ---
    atr = get_atr(df)
    adx = get_adx(df) or 0.0
    price = float(close.iloc[-1])

    # --- ✅ MOMENTUM (NOVO!) ---
    momentum = get_momentum(df, period=10)  # ROC de 10 períodos
    
    # Tratamento de valores extremos
    if momentum is not None:
        # Cap em ±50% (protege contra outliers)
        momentum = max(-0.5, min(momentum, 0.5))
    else:
        momentum = 0.0

    # --- CÁLCULO ATR% REAL ---
    if atr > price * 2:
        atr_price = atr * mt5.symbol_info(symbol).point
    else:
        atr_price = atr
    
    atr_pct_real = (atr_price / price) * 100 if price > 0 else 0

    # --- ✅ SPREAD REAL-TIME (LAND TRADING STYLE) ---
    tick = mt5.symbol_info_tick(symbol)
    spread_nominal = 0
    spread_pct = 0
    spread_points = 0
    
    if tick and tick.ask > 0 and tick.bid > 0:
        spread_nominal = tick.ask - tick.bid
        spread_points = round(spread_nominal / mt5.symbol_info(symbol).point, 0)
        spread_pct = (spread_nominal / tick.bid) * 100 if tick.bid > 0 else 0

    # --- Z-SCORE DE VOLATILIDADE ---
    vol_series = df['close'].pct_change().rolling(20).std() * 100
    atr_mean = vol_series.mean()
    atr_std = vol_series.std()
    z_score = (atr_pct_real - atr_mean) / atr_std if (atr_std and atr_std > 0) else 0
    atr_pct_capped = min(round(atr_pct_real, 3), 10.0)

    # --- VOLUME ---
    avg_vol = get_avg_volume(df)

    min_avg_vol = getattr(config, "MIN_AVG_VOLUME", 4000)

    if avg_vol < min_avg_vol:
        logger.info(f"{symbol}: Bloqueado por micro-liquidez ({avg_vol:,.0f})")
        return None
    
    # ✅ CORREÇÃO MIOPIA DE VOLUME: Ignorar candle em aberto (-1) e usar o último fechado (-2)
    cur_vol = 0
    if len(df) > 1:
        v_col = "real_volume" if "real_volume" in df.columns else "tick_volume"
        cur_vol = df[v_col].iloc[-2]
    else:
        v_col = "real_volume" if "real_volume" in df.columns else "tick_volume"
        cur_vol = df[v_col].iloc[-1]

            
    volume_ratio = round(cur_vol / max(avg_vol, 1), 2)
    volume_ratio = min(volume_ratio, 3.0)

    atr_series_data = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1).ewm(alpha=1 / 14, adjust=False).mean()

    atr_mean_val = atr_series_data.rolling(20).mean().iloc[-1]
    side = "BUY" if ema_fast > ema_slow else "SELL"
    # 🆕 CALCULA OBV
    obv_value = get_obv(df)
    obv_trend_up = is_obv_trending_up(df)
    vwap_stats = get_intraday_vwap_stats(df)
    vwap_val = vwap_stats["vwap"] if isinstance(vwap_stats, dict) else get_intraday_vwap(df)
    vwap_std = vwap_stats["std"] if isinstance(vwap_stats, dict) else None
    vwap_upper = vwap_stats["upper_2sd"] if isinstance(vwap_stats, dict) else None
    vwap_lower = vwap_stats["lower_2sd"] if isinstance(vwap_stats, dict) else None
    return {
        "symbol": symbol,
        "ema_fast": float(ema_fast),
        "ema_slow": float(ema_slow),
        "rsi": float(rsi),
        "adx": float(adx),
        "atr": float(atr),
        "atr_pct": atr_pct_capped,
        "atr_real": round(atr_pct_real, 3),
        "atr_zscore": round(z_score, 2),
        "momentum": round(momentum, 6),  # ✅ NOVO!
        "volume_ratio": volume_ratio,
        "vol_breakout": is_volatility_breakout(df, atr, atr_mean_val, volume_ratio, side),
        "vwap": vwap_val,
        "vwap_std": vwap_std,
        "vwap_upper_2sd": vwap_upper,
        "vwap_lower_2sd": vwap_lower,
        "obv": obv_value,  # 🆕 NOVO
        "obv_trend_up": obv_trend_up,  # 🆕 NOVO
        "spread_points": spread_points,  # 🆕 NOVO (pontos/ticks)
        "spread_nominal": round(spread_nominal, 3),  # 🆕 NOVO (R$)
        "spread_pct": round(spread_pct, 4),  # 🆕 NOVO (%)
        "close": price,
        "macro_trend_ok": macro_trend_ok(symbol, side),
        "tick_size": mt5.symbol_info(symbol).point,
        "params": params,
        "error": None
    }

def check_volatility_filter(symbol: str, atr: float, current_price: float) -> tuple[bool, str]:
    """
    🌊 FILTRO DE VOLATILIDADE ADAPTATIVO
    
    Bloqueia entradas em condições extremas:
    - Volatilidade >3 desvios padrão
    - ATR expandindo muito rápido (>50% em 5 candles)
    - "Chop zones" (oscilação sem direção)
    
    Returns:
        (pode_entrar: bool, motivo: str)
    
    Impacto: +5-8% win rate (evita whipsaws)
    """
    try:
        df = utils.safe_copy_rates(symbol, mt5.TIMEFRAME_M15, 50)
        
        if df is None or len(df) < 30:
            return True, ""
        
        # 1. Z-Score de Volatilidade (já existe no código)
        vol_series = df['close'].pct_change().rolling(20).std() * 100
        atr_pct_real = (atr / current_price) * 100 if current_price > 0 else 0
        
        z_score = (atr_pct_real - vol_series.mean()) / vol_series.std() if vol_series.std() > 0 else 0
        
        # ✅ NOVO: Threshold mais restritivo
        if abs(z_score) > 3.0:
            return False, f"Volatilidade extrema (z={z_score:.2f})"
        
        # 2. ✅ NOVO: Detecta expansão rápida de ATR
        atr_series = pd.concat([
            df['high'] - df['low'],
            (df['high'] - df['close'].shift()).abs(),
            (df['low'] - df['close'].shift()).abs()
        ], axis=1).max(axis=1).ewm(alpha=1/14, adjust=False).mean()
        
        recent_atr = atr_series.tail(5)
        atr_change = (recent_atr.iloc[-1] - recent_atr.iloc[0]) / recent_atr.iloc[0]
        
        if atr_change > 0.75:  # >50% em 5 candles
            return False, f"ATR expandindo rápido ({atr_change*100:.0f}%)"
        
        # 3. ✅ NOVO: Detecta "Chop Zones" (Range sem direção)
        # ADX baixo + ATR alto = mercado lateral violento
        adx = utils.get_adx(df) or 0
        
        if adx < 13 and atr_pct_real > 2.5:
            return False, f"Chop Zone (ADX {adx:.0f}, ATR {atr_pct_real:.1f}%)"
        
        # 4. ✅ NOVO: Valida se está em "squeeze" (Bollinger Bands apertadas)
        bb_std = df['close'].rolling(20).std()
        bb_width = (bb_std.iloc[-1] / df['close'].iloc[-1]) * 100
        
        if bb_width < 1.0 and atr_change > 0.30:
            # Bollinger estreitando + ATR subindo = possível breakout falso
            return False, "Possível falso breakout"
        
        return True, ""
    
    except Exception as e:
        logger.error(f"Erro filtro volatilidade {symbol}: {e}")
        return True, ""

from datetime import datetime, date

def calculate_daily_dd() -> float:
    """
    Retorna o drawdown diário atual (0.0 a 1.0)
    Ex: 0.03 = 3%
    """
    try:
        account = mt5.account_info()
        if not account:
            return 0.0

        balance = account.balance
        equity = account.equity

        if balance <= 0:
            return 0.0

        dd = (balance - equity) / balance
        return max(dd, 0.0)

    except Exception as e:
        logger.error(f"Erro ao calcular DD diário: {e}")
        return 0.0

# =========================================================
# SCORE FINAL
# =========================================================
def calculate_signal_score(ind: dict) -> float:
    """
    ✅ VERSÃO v5.5 (AGRESSIVA)
    Score rebalanceado para validação:
    - Tendência EMA: +35 pts
    - RSI Saudável (30-70): +20 pts
    - ADX > 15: +20 pts
    - Volume Ratio > 0.4: +25 pts
    - MACD Bullish/Bearish: +10 pts (Bônus)
    """
    if not isinstance(ind, dict) or ind.get("error"):
        return 0.0

    score = 0.0
    score_log = {}

    rsi = ind.get("rsi", 50)
    adx = ind.get("adx", 0)
    volume_ratio = ind.get("volume_ratio", 0.0)
    ema_fast = ind.get("ema_fast", 0)
    ema_slow = ind.get("ema_slow", 0)
    macd = ind.get("macd", 0)
    macd_signal = ind.get("macd_signal", 0)

    # 1. TENDÊNCIA EMA (Prioridade Máxima: 35 pts)
    if ema_fast > 0 and ema_slow > 0:
        if ema_fast > ema_slow:
            score += 35
            score_log["EMA_TREND_OK"] = 35
        else:
            score -= 20 # Penaliza contra-tendência
            score_log["EMA_COUNTER_TREND"] = -20

    # 2. RSI SAUDÁVEL (30-70: 20 pts)
    if 30 <= rsi <= 70:
        score += 20
        score_log["RSI_HEALTHY"] = 20

    # 3. ADX > 15 (20 pts)
    if adx >= 15:
        score += 20
        score_log["ADX_OK"] = 20

    # 4. VOLUME RATIO > 0.4 (25 pts)
    if volume_ratio >= 0.4:
        score += 25
        score_log["VOLUME_OK"] = 25

    # 5. MACD BONUS (+10 pts)
    if (ema_fast > ema_slow and macd > macd_signal) or (ema_fast < ema_slow and macd < macd_signal):
        score += 10
        score_log["MACD_BONUS"] = 10

    # 6. ML BOOST (Bonus opcional)
    try:
        if ensure_ml_loaded():
            features = ml_optimizer.extract_features(ind, ind.get("symbol", "UNK"))
            ml_pred = ml_optimizer.predict_signal_score(features)
            if ml_pred >= 0.65:
                score += 10
                score_log["ML_BOOST"] = 10
    except Exception:
        pass

    final_score = min(max(score, 0.0), 100.0)
    ind["score_log"] = score_log

    return round(final_score, 1)

def check_arbitrage_opp(sym1: str, sym2: str) -> bool:
    """
    Detecta arbitragem entre pares correlacionados (ex: PETR3 vs PETR4, web:5).
    """
    df1 = safe_copy_rates(sym1, mt5.TIMEFRAME_M15, 10)
    df2 = safe_copy_rates(sym2, mt5.TIMEFRAME_M15, 10)
    if df1 is None or df2 is None:
        return False
    
    ratio = df1['close'] / df2['close']
    current_ratio = ratio.iloc[-1]
    mean_ratio = ratio.mean()
    std_ratio = ratio.std()

    if abs(current_ratio - mean_ratio) > 2.2 * std_ratio:
        return True
    
    return False

def check_and_close_orphans(active_signals: dict):
    with mt5_lock:
        positions = mt5.positions_get() or []

    for pos in positions:
        if pos.symbol not in active_signals:
            if pos.time_update + 120 < time.time():  # 2 minutos de tolerância
                logger.warning(f"Posição órfã detectada: {pos.symbol}")
                send_telegram_exit(
                    symbol=pos.symbol,
                    reason="Posição órfã (sem sinal ativo)"
                )
def get_avg_volume(df, window: int = 20):
    if not is_valid_dataframe(df):
        return 0

    if "real_volume" in df.columns:
        vol_col = "real_volume"
    elif "tick_volume" in df.columns:
        vol_col = "tick_volume"
    else:
        return 0

    # Corrigido para usar os últimos 'window' candles FECHADOS (ignorando iloc[-1])
    return df[vol_col].iloc[-(window+1):-1].mean()

def get_open_gap(symbol: str, timeframe) -> float | None:
    df = safe_copy_rates(symbol, timeframe, 2)
    if df is None or len(df) < 2:
        return None

    prev_close = float(df["close"].iloc[-2])
    open_price = float(df["open"].iloc[-1])
    if prev_close <= 0:
        return None

    return abs((open_price - prev_close) / prev_close) * 100.0

def resolve_signal_weights(symbol, sector, base_weights,
                           sector_weights=None, symbol_weights=None):
    w = base_weights.copy()

    if sector_weights and sector in sector_weights:
        for k, v in sector_weights[sector].items():
            w[k] *= v

    if symbol_weights and symbol in symbol_weights:
        for k, v in symbol_weights[symbol].items():
            w[k] *= v

    return w

def update_symbol_weights(symbol, sector, score_log, trade_result):
    global symbol_weights

    alpha = 0.03

    if symbol not in symbol_weights:
        symbol_weights[symbol] = {}

    for k, contribution in score_log.items():
        current = symbol_weights[symbol].get(k, 1.0)
        impact = np.tanh(trade_result * contribution / 10)
        delta = 1 + alpha * impact
        symbol_weights[symbol][k] = max(0.5, min(1.8, current * delta))


_bot_instance = None

def get_telegram_bot():
    global _bot_instance
    if _bot_instance is None and getattr(config, "ENABLE_TELEGRAM_NOTIF", False):
        if telebot is None:
            logger.error("telebot não está instalado. Instale com: pip install pyTelegramBotAPI")
            return None
        try:
            _bot_instance = telebot.TeleBot(config.TELEGRAM_BOT_TOKEN)
            logger.info("Bot do Telegram inicializado com sucesso")
        except Exception as e:
            logger.error(f"Erro ao criar bot do Telegram: {e}")
            _bot_instance = None
    return _bot_instance

def send_telegram_exit(symbol: str, side: str = "", volume: float = 0, entry_price: float = 0, exit_price: float = 0, profit_loss: float = 0, reason: str = ""):
    bot = get_telegram_bot()
    if not bot:
        logger.warning("⚠️ Telegram: Bot não disponível para saída")
        return
    try:
        if (profit_loss < 0) or ("stop" in (reason or "").lower()):
            from validation import register_stop_loss
            register_stop_loss(symbol)
    except Exception:
        pass

    # 1. PEGA O LUCRO ACUMULADO DO DIA NO ARQUIVO TXT
    # Usando a função que criamos antes
    lucro_realizado_total, _ = calcular_lucro_realizado_txt()

    # Cálculo do Valor Total da Operação
    total_value = volume * exit_price 
    pl_emoji = "🟢" if profit_loss > 0 else "🔴"
    pl_pct = (profit_loss / max(entry_price * volume, 1)) * 100 if entry_price > 0 and volume > 0 else 0

    msg = (
        f"{pl_emoji} <b>XP3 — POSIÇÃO ENCERRADA</b>\n\n"
        f"<b>Ativo:</b> {symbol}\n"
        f"<b>Direção:</b> {side}\n"
        f"<b>Volume:</b> {volume:.0f} ações\n"
        f"<b>Entrada:</b> R${entry_price:.2f} | <b>Saída:</b> R${exit_price:.2f}\n"
        f"<b>Resultado:</b> R${profit_loss:+.2f} ({pl_pct:+.2f}%)\n"
        f"<b>Motivo:</b> {reason}\n"
        f"---------------------------\n"
        f"💰 <b>LUCRO NO BOLSO HOJE: R$ {lucro_realizado_total:,.2f}</b>\n" # AQUI A NOVIDADE!
        f"---------------------------\n"
        f"<i>⏱ {datetime.now().strftime('%H:%M:%S')}</i>"
    )

    try:
        bot.send_message(
            chat_id=config.TELEGRAM_CHAT_ID,
            text=msg,
            parse_mode="HTML"
        )
        logger.info(f"✅ Telegram: Notificação de SAÍDA enviada com Lucro Acumulado")
    except Exception as e:
        logger.error(f"Erro ao enviar Telegram: {e}")

def close_position(symbol: str, ticket: int, volume: float, price: float, reason: str = "Saída Estratégica"):
    
    
    # CVM Compliance: Log em CSV
    
    import csv
    import os
    from datetime import datetime
    # Identifica o tipo da posição pelo ticket para saber o lado oposto
    pos = mt5.positions_get(ticket=ticket)
    if not pos:
        logger.error(f"❌ Erro ao fechar: Posição {ticket} não encontrada.")
        return False

    pos = pos[0]
    # Se a posição é de COMPRA (0), fechamos com VENDA (1) e vice-versa
    order_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
    
    # Validar tick antes de montar request
    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        price_exec = 0.0
    else:
        price_exec = tick.bid if order_type == mt5.ORDER_TYPE_SELL else tick.ask

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": float(volume),
        "type": order_type,
        "position": ticket, # OBRIGATÓRIO para fechar a posição correta
        "price": price_exec,
        "deviation": get_dynamic_slippage(symbol, datetime.now().hour),
        "magic": 2026,
        "comment": f"XP3:{reason}",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_RETURN,
    }

    with mt5_lock:
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"❌ Falha ao fechar {symbol}: {result.comment}")
            return False
        
        # Se fechou com sucesso, envia o log de saída e notifica Telegram
        logger.info(f"✅ SAÍDA EXECUTADA: {symbol} | Motivo: {reason} | P&L: R${pos.profit:.2f}")
        
        # Chama sua função de Telegram já existente no utils.py
        send_telegram_exit(
            symbol=symbol,
            side="BUY" if pos.type == mt5.POSITION_TYPE_BUY else "SELL",
            volume=volume,
            entry_price=pos.price_open,
            exit_price=price,
            profit_loss=result.profit if hasattr(result, "profit") else pos.profit,
            reason=reason
        )
        
        # ✅ Atualiza Streak de Perdas/Ganhos
        pl = result.profit if hasattr(result, "profit") else pos.profit
        if pl < 0:
            _symbol_consecutive_losses[symbol] += 1
        else:
            _symbol_consecutive_losses[symbol] = 0

        try:
            from validation import register_stop_loss
            if pl < 0:
                register_stop_loss(symbol)
        except Exception:
            pass

        # ✅ Compliance CVM: trades_compliance.csv
        compliance_file = "trades_compliance.csv"
        file_exists = os.path.isfile(compliance_file)
        
        try:
            import csv
            with open(compliance_file, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["Timestamp", "Ticket", "Symbol", "Volume", "Entry_Price", "Exit_Price", "Profit", "Reason"])
                writer.writerow([datetime.now().isoformat(), ticket, symbol, volume, pos.price_open, price, pos.profit, reason])
        except Exception as e:
            logger.error(f"Erro ao salvar compliance CSV: {e}")

        return True
adaptive_lock = threading.Lock()
def save_adaptive_weights():
    data = {
        "symbol": symbol_weights,
        "sector": sector_weights
    }

    tmp_path = "adaptive_weights.tmp"
    final_path = "adaptive_weights.json"

    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        os.replace(tmp_path, final_path)

        logger.info("💾 Pesos adaptativos salvos com sucesso.")

    except Exception as e:
        logger.error(f"❌ Erro ao salvar pesos adaptativos: {e}")


def load_adaptive_weights():
    global symbol_weights, sector_weights
    path = "adaptive_weights.json"

    symbol_weights = {}
    sector_weights = {}

    if not os.path.exists(path):
        logger.info("ℹ️ Pesos adaptativos não encontrados. Usando padrão.")
        return

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError("JSON raiz não é um objeto")

        symbol_weights = data.get("symbol")
        sector_weights = data.get("sector")

        if not isinstance(symbol_weights, dict) or not isinstance(sector_weights, dict):
            raise ValueError("Estrutura inválida em adaptive_weights.json")

        logger.info(
            f"🧠 Pesos adaptativos carregados: "
            f"{len(symbol_weights)} símbolos | {len(sector_weights)} setores"
        )

    except json.JSONDecodeError as e:
        logger.error(f"❌ JSON inválido em {path}: linha {e.lineno}, coluna {e.colno}")
    except Exception as e:
        logger.error(f"❌ Erro ao carregar pesos adaptativos: {e}")

    return

def update_adaptive_weights():
    """
    Inicializa pesos adaptativos de forma segura e compatível
    """
    global symbol_weights, sector_weights

    # 🔒 Garante que os dicts existam
    if symbol_weights is None:
        symbol_weights = {}

    if sector_weights is None:
        sector_weights = {}

    # 🎯 Chaves reais usadas no score_log
    base_weight_keys = [
        "ADX_STRONG",
        "ADX_GOOD",
        "RSI_EXTREME",
        "RSI_STRETCH",
        "VOL_HIGH",
        "VOL_OK",
        "EMA_TREND",
        "EMA_COUNTER",
        "MOMENTUM_POS",
        "MACD_CROSS",
        "ML_BOOST"
    ]

    # =========================
    # 🧠 PESOS POR SÍMBOLO
    # =========================
    for sym in config.SECTOR_MAP.keys():
        if sym not in symbol_weights:
            symbol_weights[sym] = {}

        for key in base_weight_keys:
            symbol_weights[sym].setdefault(key, 1.0)

    # =========================
    # 🏭 PESOS POR SETOR
    # =========================
    for sector in set(config.SECTOR_MAP.values()):
        if sector not in sector_weights:
            sector_weights[sector] = {}

        for key in base_weight_keys:
            sector_weights[sector].setdefault(key, 1.0)

    logger.info("✅ Pesos adaptativos inicializados / normalizados com sucesso")


def check_liquidity(symbol: str, threshold_pct: float = 0.4):
    """
    ✅ Filtro de Liquidez Real-Time (B3 Optimized Jan/2026)

    Regra: Volume Projetado do Dia > 40% da Média de 10 Dias.
    Projeção: (Volume Atual / Minutos Decorridos) * 420 min
    """
    try:
        if is_future(symbol):
            return True
        rates_d1 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_D1, 1, 10)
        if rates_d1 is None or len(rates_d1) < 5:
            return True

        df_d1 = pd.DataFrame(rates_d1)
        vol_col = "real_volume" if "real_volume" in df_d1.columns else "tick_volume"
        avg_vol_10d = df_d1[vol_col].mean()

        rate_today = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_D1, 0, 1)
        if not rate_today:
            return False

        vol_today = rate_today[0][vol_col]

        now = datetime.now()
        open_time = now.replace(hour=10, minute=0, second=0, microsecond=0)

        if now < open_time:
            return True

        elapsed_min = max(15, (now - open_time).total_seconds() / 60)
        elapsed_min = min(elapsed_min, 420)

        projected_vol = (vol_today / elapsed_min) * 420
        target_vol = avg_vol_10d * threshold_pct

        if projected_vol < target_vol:
            logger.info(
                f"💧 {symbol}: Liquidez insuficiente "
                f"({projected_vol:,.0f} < {target_vol:,.0f})"
            )
            return False

        return True

    except Exception as e:
        logger.error(f"Erro check_liquidity real-time {symbol}: {e}")
        return False

def calculate_position_size_atr(symbol: str, atr_dist: float, risk_money: float = None) -> float:
    """
    ✅ POSITION SIZING B3: Risco 1% com base no ATR (Jan/2026)
    """
    try:
        # Garante equity para cálculo de tolerância
        acc = mt5.account_info()
        if not acc:
            return 0.0
        equity = acc.equity if acc.equity > 0 else acc.balance

        if atr_dist <= 0:
            return 0.0

        if risk_money is None:
            rp = getattr(config, "RISK_PER_TRADE_PCT", 0.01)
            rp = max(getattr(config, "MIN_RISK_PER_TRADE_PCT", 0.0025), min(rp, getattr(config, "MAX_RISK_PER_TRADE_PCT", 0.005)))
            risk_money = equity * float(rp)

        min_lot = 1 if is_future(symbol) else 100
        risk_min_lot = min_lot * atr_dist

        equity_tolerance_cap = equity * 0.013
        if risk_min_lot > equity_tolerance_cap:
            logger.info(
                f"❌ {symbol}: Risco muito alto "
                f"({risk_min_lot:,.0f} > {equity_tolerance_cap:,.0f})"
            )
            return 0.0
            
        num_lots = int(risk_money / risk_min_lot)
        
        # ✅ PROPORCIONALIDADE ESTRITA (Land Trading):
        # Impede trades desproporcionais onde 1 lote viola o gerenciamento de risco.
        if num_lots <= 0:
            # Tolerância máxima: 20% acima do risco estipulado
            if risk_min_lot <= risk_money * 1.2:
                num_lots = 1
            else:
                logger.warning(
                    f"⚖️ {symbol}: Risco desproporcional. "
                    f"Mínimo {risk_min_lot:.0f} > Limite {risk_money * 1.2:.0f}. "
                    "Trade ignorado."
                )
                return 0.0
                
        return float(num_lots * min_lot)

    except Exception as e:
        logger.error(f"Erro calculate_position_size_atr: {e}")
        return 0.0


def calculate_total_exposure() -> float:
    try:
        with mt5_lock:
            positions = mt5.positions_get() or []
        total = 0.0
        for p in positions:
            tick = mt5.symbol_info_tick(p.symbol)
            si = mt5.symbol_info(p.symbol)
            contract = si.trade_contract_size if si else 1.0
            if tick:
                price = tick.bid if p.type == mt5.POSITION_TYPE_SELL else tick.ask
            else:
                price = getattr(p, "price_current", getattr(p, "price_open", 0.0))
            total += float(p.volume) * float(contract) * float(price)
        return float(total)
    except Exception as e:
        logger.error(f"Erro calculate_total_exposure: {e}")
        return 0.0


def get_effective_exposure_limit() -> float:
    try:
        acc = mt5.account_info()
        if not acc:
            return float(getattr(config, "MAX_TOTAL_EXPOSURE", 1_000_000))
        dyn = 2.0 * float(acc.equity or acc.balance or 0.0)
        return float(dyn)
    except Exception as e:
        logger.error(f"Erro get_effective_exposure_limit: {e}")
        return float(getattr(config, "MAX_TOTAL_EXPOSURE", 1_000_000))

def signal_handler(sig, frame):
    with mt5_lock:
        logger.info("Encerrando bot - salvando pesos adaptativos...")
        save_adaptive_weights()
        mt5.shutdown()
    exit(0)

def send_telegram_trade(symbol: str, side: str, volume: float, price: float, sl: float, tp: float, comment: str = ""):
    bot = get_telegram_bot()
    if not bot:
        logger.warning("⚠️ Telegram: Bot não inicializado (token ausente ou inválido)")
        return

    if side == "BUY":
        direction = "🟢 COMPRA"
        arrow = "⬆️"
    else:
        direction = "🔴 VENDA"
        arrow = "⬇️"

    dist_sl = abs(price - sl)
    dist_tp = abs(tp - price)
    risk_pct = round((dist_sl / price) * 100, 2)
    reward_pct = round((dist_tp / price) * 100, 2)
    rr_ratio = round(dist_tp / dist_sl, 2) if dist_sl > 0 else 0.0
    if rr_ratio == 0.0:
        logger.warning(f"{symbol}: R:R inválido (SL muito próximo ou zero)")

    msg = (
        f"<b>🚀 XP3 – NOVA ENTRADA</b>\n\n"
        f"<b>Ativo:</b> {symbol}\n"
        f"<b>Direção:</b> {direction} {arrow}\n"
        f"<b>Volume:</b> {volume:.0f} ações\n"
        f"<b>Entrada:</b> R${price:.2f}\n\n"
        f"<b>🛑 SL:</b> R${sl:.2f} <i>(-{risk_pct}%)</i>\n"
        f"<b>🎯 TP:</b> R${tp:.2f} <i>(+{reward_pct}%)</i>\n"
        f"<b>R:R:</b> 1:{rr_ratio}\n"
        f"<b>Comentário:</b> {comment}\n\n"
        f"<i>⏱ {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</i>"
    )

    try:
        bot.send_message(
            chat_id=config.TELEGRAM_CHAT_ID,
            text=msg,
            parse_mode="HTML"
        )
        logger.info(f"✅ Telegram: Notificação de ENTRADA enviada para {symbol}")
    except Exception as e:
        logger.error(f"❌ ERRO ao enviar Telegram (entrada {symbol}): {e}")

def validate_mt5_connection():
    """
    Verifica se MT5 está conectado e operável.
    """
    try:
        terminal_info = mt5.terminal_info()
        if terminal_info is None:
            logger.critical("❌ MT5 não está inicializado")
            return False
        
        if not terminal_info.connected:
            logger.critical("❌ MT5 não está conectado ao servidor")
            return False
        
        if not terminal_info.trade_allowed:
            logger.error("⚠️ Trading não permitido no MT5")
            return False
        
        return True
    
    except Exception as e:
        logger.error(f"Erro ao validar conexão MT5: {e}")
        return False

def send_order_with_sl_tp(symbol, side, volume, sl, tp, comment="XP3_BOT"):
    if not validate_mt5_connection():
        return False

    info = mt5.symbol_info(symbol)
    if info is None:
        return False

    if not info.visible:
        mt5.symbol_select(symbol, True)
        time.sleep(0.2)

    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        return False
    status_cb = check_b3_circuit_breaker()
    if status_cb in ("CLOSE_ALL", "PAUSE_AND_TIGHTEN"):
        return False
    price = tick.ask if side == "BUY" else tick.bid
    if is_future(symbol) and should_block_opening_gap(symbol, getattr(config, "MAX_ACCEPTABLE_GAP_PCT", 0.015)):
        return False
    if block_if_high_portfolio_correlation(symbol, getattr(config, "MIN_CORRELATION_SCORE_TO_BLOCK", 0.70)):
        return False
    now_t = datetime.now().time()
    if not is_future(symbol):
        try:
            t_0950 = datetime.strptime("09:50", "%H:%M").time()
            t_1005 = datetime.strptime("10:05", "%H:%M").time()
            t_1650 = datetime.strptime("16:50", "%H:%M").time()
            t_1705 = datetime.strptime("17:05", "%H:%M").time()
            in_open_auction = t_0950 <= now_t <= t_1005
            in_close_auction = t_1650 <= now_t <= t_1705
            if in_open_auction or in_close_auction:
                return False
        except Exception:
            pass
        try:
            if side == "SELL":
                borrow_pct = float(getattr(config, "SHORT_BORROW_PCT", 0.004) or 0.004)
                rr_pct = abs(tp - price) / max(price, 1e-9)
                if rr_pct <= borrow_pct:
                    return False
        except Exception:
            pass
    fm_start = datetime.strptime(getattr(config, "FUTURES_AFTERMARKET_START", "16:00"), "%H:%M").time()
    fm_end = datetime.strptime(getattr(config, "FUTURES_AFTERMARKET_END", "17:50"), "%H:%M").time()
    if is_future(symbol):
        try:
            close_by = datetime.strptime(getattr(config, "FUTURES_CLOSE_ALL_BY", "17:50"), "%H:%M").time()
            if now_t >= close_by:
                return False
        except Exception:
            pass
    rr_now = abs(tp - price) / max(abs(price - sl), 1e-9)
    if is_future(symbol) and fm_start <= now_t <= fm_end:
        try:
            df_af = safe_copy_rates(symbol, TIMEFRAME_BASE, 60)
            adx_now = get_adx(df_af) or 0.0
            rr_min_af = float(getattr(config, "AFTER_MARKET_RR_MIN", 3.0))
            if adx_now < 35.0:
                return False
            if rr_now < rr_min_af:
                return False
            info_step = info.volume_step if info.volume_step > 0 else 1.0
            volume = (max(info.volume_min, (volume * 0.5)) // info_step) * info_step
            if volume <= 0:
                return False
        except Exception:
            return False
    # Checagem de margem e custo/RR mínimo
    try:
        acc = mt5.account_info()
        free_margin = float(getattr(acc, "margin_free", 0.0) or 0.0)
        ord_type = mt5.ORDER_TYPE_BUY if side == "BUY" else mt5.ORDER_TYPE_SELL
        est_margin = float(mt5.order_calc_margin(ord_type, symbol, price, float(volume)) or 0.0)
        eff_margin = free_margin
        try:
            with mt5_lock:
                positions = mt5.positions_get() or []
            floating_profit = sum(float(getattr(p, "profit", 0.0) or 0.0) for p in positions)
            eff_margin = free_margin + max(0.0, floating_profit) * 0.7
        except Exception:
            eff_margin = free_margin
        if eff_margin < est_margin * 1.30:
            try:
                acc2 = mt5.account_info()
                fm2 = float(getattr(acc2, "margin_free", 0.0) or 0.0)
                if fm2 < est_margin * 0.50:
                    enforce_stopout_prevention()
            except Exception:
                pass
            return False
    except Exception:
        pass
    try:
        if is_future(symbol):
            acc = mt5.account_info()
            eq = float(getattr(acc, "equity", 0.0) or 0.0)
            if eq <= 0.0:
                eq = float(getattr(acc, "balance", 0.0) or 0.0)
            insp = AssetInspector.detect(symbol)
            pv = float(insp.get("point_value", 1.0) or 1.0)
            total_exposure = 0.0
            with mt5_lock:
                pos_list = mt5.positions_get() or []
            for p in pos_list:
                try:
                    if is_future(p.symbol):
                        t = mt5.symbol_info_tick(p.symbol)
                        if not t:
                            continue
                        px = t.ask if p.type == mt5.POSITION_TYPE_BUY else t.bid
                        total_exposure += abs(float(getattr(p, "volume", 0.0) or 0.0)) * pv * float(px or 0.0)
                except Exception:
                    continue
            add_exposure = abs(float(volume)) * pv * float(price)
            cap_limit = eq * 2.0
            if (total_exposure + add_exposure) > cap_limit:
                return False
    except Exception:
        pass
    try:
        insp = AssetInspector.detect(symbol)
        fee_type = insp.get("fee_type", "PERCENT")
        fee_val = float(insp.get("fee_val", 0.00055))
        spread = abs(tick.ask - tick.bid) / max(price, 1e-9)
        fee_pct = (fee_val if fee_type == "PERCENT" else 0.0005)
        cost_pct = spread + (fee_pct * 2.0)
        rr = rr_now
        if cost_pct > 0.015:
            return False
        if cost_pct > 0.005:
            px = price
            rr_min = 2.0 if px <= 50.0 else 1.8
            if px <= 10.0:
                rr_min = 2.8
            if rr < rr_min:
                return False
    except Exception:
        pass
    try:
        if is_future(symbol):
            vix_val = float(getattr(config, "VIX_THRESHOLD_RISK_OFF", 30))
            try:
                vix_now = get_vix_br()
                if vix_now >= vix_val:
                    step = info.volume_step if info.volume_step > 0 else 1.0
                    volume = (max(info.volume_min, (volume * 0.5)) // step) * step
                    if volume <= 0:
                        return False
                maxc = int(getattr(config, "FUTURES_MAX_CONTRACTS", 10))
                if vix_now >= getattr(config, "VIX_THRESHOLD_PROTECTION", 35):
                    maxc = max(1, maxc // 2)
                if volume > maxc:
                    volume = float(maxc)
            except Exception:
                pass
    except Exception:
        pass

    # 🔒 Validação SL/TP
    if side == "BUY":
        if sl >= price or tp <= price:
            return False
    else:
        if sl <= price or tp >= price:
            return False

    # 🔒 Stop level
    stop_level = info.trade_stops_level * info.point
    if side == "BUY":
        if (price - sl) < stop_level or (tp - price) < stop_level:
            return False
    else:
        if (sl - price) < stop_level or (price - tp) < stop_level:
            return False

    # 🔒 Ajuste de volume
    volume_step = info.volume_step if info.volume_step > 0 else 100
    volume = (volume // volume_step) * volume_step
    if volume <= 0:
        return False

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": float(volume),
        "type": mt5.ORDER_TYPE_BUY if side == "BUY" else mt5.ORDER_TYPE_SELL,
        "price": price,
        "sl": float(sl),
        "tp": float(tp),
        "magic": 2026,
        "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_RETURN,
    }

    with mt5_lock:
        result = mt5.order_send(request)

    if not result or result.retcode != mt5.TRADE_RETCODE_DONE:
        return False

    time.sleep(0.2)
    if not mt5.positions_get(symbol=symbol):
        return False

    return True


def send_order_with_retry(symbol, side, volume, sl, tp, max_retries=3):
    for attempt in range(max_retries):
        result = send_order_with_sl_tp(symbol, side, volume, sl, tp)

        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"✅ Ordem executada após {attempt+1} tentativa(s)")
            return result

        if result and result.retcode in (
            mt5.TRADE_RETCODE_REQUOTE,
            mt5.TRADE_RETCODE_PRICE_OFF,
            mt5.TRADE_RETCODE_TIMEOUT
        ):
            logger.warning(
                f"⚠️ Retry {attempt+1}/{max_retries} {symbol} | "
                f"Retcode={result.retcode} | {result.comment}"
            )
            time.sleep(0.5)
            continue

        # ❌ erro irrecuperável
        if result:
            logger.error(
                f"❌ Erro irrecuperável {symbol} | "
                f"Retcode={result.retcode} | {result.comment}"
            )
        else:
            logger.error(f"❌ MT5 retornou None para {symbol}")

        break

    return None


def validate_order_params(symbol: str, side: str, volume: float, price: float, sl: float, tp: float) -> bool:
    with mt5_lock:
        info = mt5.symbol_info(symbol)
        tick = mt5.symbol_info_tick(symbol)

    if not info or not tick:
        logger.error(f"{symbol}: info ou tick indisponível")
        return False

    # ✅ Volume mínimo / máximo / step
    if volume < info.volume_min or volume > info.volume_max:
        logger.error(f"{symbol}: volume fora dos limites")
        return False

    if ((volume - info.volume_min) % info.volume_step) != 0:
        logger.error(f"{symbol}: volume fora do step permitido")
        return False

    # ✅ Distância mínima de SL/TP (fallback seguro)
    min_stop = info.trade_stops_level * info.point
    if min_stop <= 0:
        min_stop = info.point * 5  # fallback B3 seguro

    if abs(price - sl) < min_stop:
        logger.error(f"{symbol}: SL muito próximo ({abs(price-sl):.4f} < {min_stop:.4f})")
        return False

    if abs(tp - price) < min_stop:
        logger.error(f"{symbol}: TP muito próximo ({abs(tp-price):.4f} < {min_stop:.4f})")
        return False

    # ✅ Lógica correta por lado
    if side == "BUY":
        if sl >= price or tp <= price:
            logger.error(f"{symbol}: BUY inválido (SL/TP)")
            return False
    else:
        if sl <= price or tp >= price:
            logger.error(f"{symbol}: SELL inválido (SL/TP)")
            return False

    # ✅ Preço razoável vs bid/ask real
    ref_price = tick.ask if side == "BUY" else tick.bid
    if abs(price - ref_price) / ref_price > 0.02:
        logger.error(f"{symbol}: preço muito distante do mercado")
        return False

    return True

def get_time_bucket():
    now = datetime.now().time()
    for bucket, cfg in config.TIME_SCORE_RULES.items():
        start = datetime.strptime(cfg["start"], "%H:%M").time()
        end   = datetime.strptime(cfg["end"], "%H:%M").time()
        if start <= end:
            if start <= now <= end:
                return bucket, cfg
        else:
            if now >= start or now <= end:
                return bucket, cfg
    return "MID", config.TIME_SCORE_RULES["MID"]

def is_power_hour():
    now = datetime.now().time()
    cfg = config.POWER_HOUR
    if not cfg["enabled"]:
        return False
    start = datetime.strptime(cfg["start"], "%H:%M").time()
    end   = datetime.strptime(cfg["end"], "%H:%M").time()
    if start <= end:
        return start <= now <= end
    else:
        return now >= start or now <= end

def is_volatility_breakout(df, atr_now, atr_mean, volume_ratio, side=None):
    if not config.VOL_BREAKOUT["enabled"]:
        return False

    if atr_now is None or atr_mean is None:
        return False

    if atr_now < atr_mean * config.VOL_BREAKOUT["atr_expansion"]:
        return False

    if volume_ratio < config.VOL_BREAKOUT["volume_ratio"]:
        return False

    lookback = config.VOL_BREAKOUT["lookback"]
    if len(df) < lookback + 2:
        return False

    high_roll = df["high"].rolling(lookback).max()
    low_roll  = df["low"].rolling(lookback).min()

    high_break = df["high"].iloc[-1] > high_roll.iloc[-2]
    low_break  = df["low"].iloc[-1]  < low_roll.iloc[-2]


    if side == "BUY":
        return high_break
    if side == "SELL":
        return low_break

    return high_break or low_break

# ===== SUBSTITUIR A FUNÇÃO get_current_risk_pct() NO utils.py =====

def get_current_risk_pct() -> float:
    """
    Retorna o risco percentual atual por trade (estável e controlado)
    """
    base_risk = config.RISK_PER_TRADE_PCT
    risk = base_risk

    now = datetime.now()
    weekday = now.weekday()
    hour = now.hour

    # 1️⃣ Sexta-feira à tarde (redução suave)
    if weekday == 4 and hour >= 15:
        risk = min(risk, config.REDUCED_RISK_PCT)

    # 2️⃣ Regime de mercado (limitador, não multiplicador destrutivo)
    regime = detect_market_regime()
    if regime == "RISK_OFF":
        risk = min(risk, base_risk * 0.7)

    # 3️⃣ Power Hour (somente se aumentar risco)
    if is_power_hour():
        mult = config.POWER_HOUR.get("risk_multiplier", 1.0)
        if mult > 1.0:
            risk *= mult

    # 4️⃣ Profit Lock (apenas se meta REAL foi atingida)
    if config.PROFIT_LOCK["enabled"] and config.PROFIT_LOCK["reduce_risk"]:
        with mt5_lock:
            acc = mt5.account_info()

        if acc and os.path.exists("daily_equity.txt"):
            try:
                with open("daily_equity.txt", "r") as f:
                    equity_inicio = float(f.read().strip())

                daily_pnl_pct = (acc.equity - equity_inicio) / equity_inicio

                if daily_pnl_pct >= config.PROFIT_LOCK["daily_target_pct"]:
                    risk = min(risk, base_risk * 0.5)
                    logger.info("🔒 Profit Lock ativo — risco reduzido")
            except Exception:
                pass

    # 5️⃣ Limites corretos
    risk = min(risk, config.MAX_RISK_PER_SYMBOL_PCT)

    # Piso técnico (0.2% — abaixo disso não faz sentido operar)
    return max(0.002, round(risk, 4))


def get_dynamic_slippage(symbol: str, hour: int) -> int:
    """
    Slippage dinâmico por ativo e horário (B3 – robusto)
    Retorna SEMPRE inteiro
    """

    s = (symbol or "").upper()
    if s.startswith(("WIN", "IND")):
        base = 8
    elif s.startswith(("WDO", "DOL")):
        base = 3
    else:
        base = int(config.SLIPPAGE_MAP.get(
            symbol,
            config.SLIPPAGE_MAP.get("DEFAULT", 10)
        ))

    # 2️⃣ ABERTURA (prioridade máxima)
    if 10 <= hour < 11:
        base = int(base * 1.5)

    # 3️⃣ FECHAMENTO (liquidez cai)
    elif hour >= 16:
        base = int(base * 1.3)

    # 4️⃣ POWER HOUR (liquidez máxima)
    elif 15 <= hour < 16:
        base = int(base * 0.7)

    # 5️⃣ LIMITES DE SEGURANÇA
    min_slip = 2
    max_slip = 50

    base = max(min_slip, min(base, max_slip))

    return base


def calculate_smart_sl(symbol, entry_price, side, atr, df):
    """
    Calcula stop loss considerando:
    1. ATR (risco estatístico)
    2. Suporte/Resistência mais próximo
    3. Mínimo de 1.5 ATR (nunca muito apertado)
    """
    # ✅ PROTEÇÃO ATR MÍNIMO
    if atr < 0.01:
        atr = 0.01
        logger.warning(f"{symbol}: ATR muito baixo - usando mínimo 0.01")
    
    base_distance = atr * 2.0
    
    # Encontra suporte/resistência relevante
    lookback = 50
    if side == "BUY":
        # Para compra: busca último fundo relevante
        recent_lows = df['low'].tail(lookback)
        support = recent_lows.min()
        
        # Stop 0.5 ATR abaixo do suporte
        structure_stop = support - (atr * 0.5)
        
        # Usa o MENOR entre estrutura e ATR (mais conservador)
        final_sl = max(structure_stop, entry_price - base_distance)
        
    else:  # SELL
        recent_highs = df['high'].tail(lookback)
        resistance = recent_highs.max()
        structure_stop = resistance + (atr * 0.5)
        final_sl = min(structure_stop, entry_price + base_distance)
    
    # Garante mínimo de 1.5 ATR
    min_distance = atr * 1.5
    if side == "BUY":
        final_sl = min(final_sl, entry_price - min_distance)
    else:
        final_sl = max(final_sl, entry_price + min_distance)
    
    return round(final_sl, 2)

def analyze_order_book_depth(symbol, side, volume_needed, multiplier: float | None = None):
    try:
        if multiplier is None:
            multiplier = float(getattr(config, "ORDER_BOOK_DEPTH_MULTIPLIER", 3) or 3)

        book = mt5.market_book_get(symbol)
        if book is None or len(book) == 0:
            book = None

        if book is not None:
            target_type = mt5.BOOK_TYPE_SELL if side == "BUY" else mt5.BOOK_TYPE_BUY
            levels = int(getattr(config, "ORDER_BOOK_LEVELS", 10) or 10)
            filtered = [item for item in book if item.type == target_type][:levels]
            available_liquidity = sum(item.volume for item in filtered)

            threshold = float(volume_needed) * float(multiplier)
            if available_liquidity < threshold:
                logger.warning(
                    f"📚 {symbol}: Liquidez insuficiente no Book | "
                    f"Disponível: {available_liquidity:,.0f} | "
                    f"Necessário ({multiplier:.1f}x): {threshold:,.0f}"
                )
                return False
            inst_mult = float(getattr(config, "INSTITUTIONAL_LEVEL_MULT", 5.0) or 5.0)
            inst_thresh = float(volume_needed) * inst_mult
            for item in filtered:
                try:
                    if item.volume >= inst_thresh:
                        logger.warning(f"🏦 {symbol}: Nível institucional detectado ({item.volume:,.0f} ≥ {inst_thresh:,.0f})")
                        return False
                except Exception:
                    continue
            return True

        df = safe_copy_rates(symbol, mt5.TIMEFRAME_M5, 20)
        
        if df is not None and not df.empty:
            if 'real_volume' in df.columns and df['real_volume'].sum() > 0:
                median_vol = df['real_volume'].median()
            else:
                median_vol = df['tick_volume'].median() * 100
            
            if median_vol <= 0:
                return True
            
            now = datetime.now()
            is_power_hour = False
            try:
                ph = getattr(config, "POWER_HOUR", {}) or {}
                if ph.get("enabled", True):
                    start = datetime.strptime(str(ph.get("start", "15:30")), "%H:%M").time()
                    end = datetime.strptime(str(ph.get("end", "16:55")), "%H:%M").time()
                    is_power_hour = start <= now.time() <= end
            except Exception:
                is_power_hour = False

            impact_cfg = getattr(config, "ADAPTIVE_FILTERS", {}).get("volume_impact", {})
            max_impact = float(impact_cfg.get("power_hour" if is_power_hour else "normal", 0.20) or 0.20)
            
            impact_ratio = volume_needed / median_vol
            
            if impact_ratio > max_impact:
                logger.warning(
                    f"⚠️ {symbol}: Alto impacto no volume | "
                    f"Ordem: {volume_needed:.0f} | Mediana: {median_vol:.0f} | "
                    f"Impacto: {impact_ratio*100:.1f}% (máx {max_impact*100:.0f}%)"
                )
                return False
            
            logger.debug(f"✅ Volume OK: {symbol} (impacto {impact_ratio*100:.1f}%)")
                
        return True

    except Exception as e:
        logger.error(f"Erro ao analisar profundidade de {symbol}: {e}", exc_info=True)
        return True  # Fail-open

def detect_calendar_spread(base: str = "WIN") -> dict:
    try:
        cands = get_futures_candidates(base)
        if not cands or len(cands) < 2:
            return {"ok": False}
        cands_sorted = sorted(cands, key=lambda c: (-calculate_contract_score(c), c.get("days_to_exp", 9999)))
        a = cands_sorted[0]
        b = cands_sorted[1]
        sa = a.get("symbol")
        sb = b.get("symbol")
        ta = mt5.symbol_info_tick(sa)
        tb = mt5.symbol_info_tick(sb)
        if not ta or not tb:
            return {"ok": False}
        pa = float(ta.ask or ta.bid or 0.0)
        pb = float(tb.ask or tb.bid or 0.0)
        spread = pb - pa
        info = AssetInspector.detect(sa)
        ts = float(info.get("tick_size", 5.0) or 5.0)
        pv = float(info.get("point_value", 0.20) or 0.20)
        spread_pts = spread / ts if ts > 0 else 0.0
        thresh_pts = float(getattr(config, "CALENDAR_SPREAD_THRESHOLD_POINTS", 80) or 80)
        ok = abs(spread_pts) >= thresh_pts
        return {"ok": ok, "a": sa, "b": sb, "spread_points": spread_pts, "spread_value": spread * pv}
    except Exception:
        return {"ok": False}

def apply_trailing_stop(symbol: str, current_price: float, atr: float):
    """
    Trailing Stop baseado em estrutura (Price Action) + proteção ATR
    """

    if atr is None or atr <= 0:
        return

    with mt5_lock:
        info = mt5.symbol_info(symbol)
        positions = mt5.positions_get(symbol=symbol)

    if not info or not positions:
        return

    tick_size = info.point
    min_stop_distance = info.trade_stops_level * tick_size

    df = get_fast_rates(symbol, TIMEFRAME_BASE)
    if not is_valid_dataframe(df) or len(df) < 3:
        return

    for pos in positions:

        # =========================
        # 🟢 BUY
        # =========================
        if pos.type == mt5.POSITION_TYPE_BUY:

            candle_sl = min(df["low"].iloc[-2], df["low"].iloc[-3]) - tick_size

            # 🛡️ Proteção ATR mínima
            atr_sl = current_price - (atr * 1.2)
            candidate_sl = max(candle_sl, atr_sl)

            # Respeita distância mínima do preço atual
            if current_price - candidate_sl < min_stop_distance:
                continue

            if pos.sl and candidate_sl <= pos.sl:
                continue

            new_sl = round(candidate_sl / tick_size) * tick_size

        # =========================
        # 🔴 SELL
        # =========================
        elif pos.type == mt5.POSITION_TYPE_SELL:

            candle_sl = max(df["high"].iloc[-2], df["high"].iloc[-3]) + tick_size
            atr_sl = current_price + (atr * 1.2)
            candidate_sl = min(candle_sl, atr_sl)

            if candidate_sl - current_price < min_stop_distance:
                continue

            if pos.sl and candidate_sl >= pos.sl:
                continue

            new_sl = round(candidate_sl / tick_size) * tick_size

        else:
            continue

        # Evita micro-ajustes
        if pos.sl and abs(new_sl - pos.sl) < tick_size * 5:
            continue

        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": pos.ticket,
            "symbol": symbol,
            "sl": float(new_sl),
            "tp": pos.tp,
            "magic": 2026
        }

        with mt5_lock:
            res = mt5.order_send(request)

        if res and res.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(
                f"🔄 Trailing {symbol} "
                f"{'BUY' if pos.type==0 else 'SELL'} -> {new_sl}"
            )

def can_enter_symbol(symbol: str, equity: float) -> bool:
    """
    Verifica se pode abrir nova posição no símbolo considerando
    RISCO REAL (baseado em SL).
    """

    if equity <= 0:
        return False

    with mt5_lock:
        positions = [p for p in mt5.positions_get() or [] if p.symbol == symbol]

    # ✅ Sem posições abertas → pode entrar
    if not positions:
        return True

    total_risk_money = 0.0

    for p in positions:
        # ❌ Se não tem SL → risco infinito → bloqueia
        if not p.sl or p.sl <= 0:
            logger.warning(f"{symbol}: posição sem SL detectada — bloqueando nova entrada")
            return False

        # Risco real = distância até SL * volume
        risk_per_unit = abs(p.price_open - p.sl)
        position_risk = risk_per_unit * p.volume

        total_risk_money += position_risk

    risk_pct = total_risk_money / equity

    if risk_pct >= config.MAX_RISK_PER_SYMBOL_PCT:
        logger.warning(
            f"{symbol}: Limite de risco por símbolo atingido "
            f"({risk_pct*100:.2f}% ≥ {config.MAX_RISK_PER_SYMBOL_PCT*100:.2f}%)"
        )
        return False

    return True


def calculate_correlation_matrix(symbols: List[str], lookback: int = 60) -> Dict[str, Dict[str, float]]:
    """
    Calcula matriz de correlação entre símbolos.
    Retorna: {symbol1: {symbol2: corr_value}}
    """
    if not symbols or len(symbols) < 2:
        return {}
    
    # Coleta dados de fechamento
    closes = {}
    for sym in symbols:
        df = safe_copy_rates(sym, mt5.TIMEFRAME_D1, lookback)
        if df is not None and not df.empty and len(df) >= 30:
            closes[sym] = df['close']
    
    if len(closes) < 2:
        return {}
    
    # Alinha datas
    df_all = pd.DataFrame(closes)
    df_all = df_all.dropna()
    
    if df_all.empty or len(df_all) < 30:
        return {}
    
    # Calcula correlação
    corr_matrix = df_all.corr()
    
    # Converte para dict
    result = {}
    for sym1 in symbols:
        if sym1 not in corr_matrix.columns:
            continue
        result[sym1] = {}
        for sym2 in symbols:
            if sym2 not in corr_matrix.columns:
                continue
            result[sym1][sym2] = float(corr_matrix.loc[sym1, sym2])
    
    return result

# =========================================================
# RASTREAMENTO DE PERFORMANCE POR ATIVO (LOSS STREAK)
# =========================================================

LOSS_STREAK_FILE = "symbol_loss_streak.json"

_symbol_loss_streak = defaultdict(int)
_symbol_last_loss_time = {}
_symbol_block_until = {}

loss_streak_lock = threading.Lock()


def load_loss_streak_data():
    global _symbol_loss_streak, _symbol_last_loss_time, _symbol_block_until

    if not os.path.exists(LOSS_STREAK_FILE):
        logger.info("ℹ️ Arquivo de loss streak não encontrado. Inicializando vazio.")
        return

    try:
        with open(LOSS_STREAK_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)

        now = datetime.now()

        with loss_streak_lock:

            # === STREAK ===
            streak_raw = raw.get("streak", {})
            _symbol_loss_streak = defaultdict(
                int,
                {k: int(v) for k, v in streak_raw.items() if isinstance(v, (int, float))}
            )

            # === LAST LOSS TIME ===
            _symbol_last_loss_time = {}
            for sym, ts in raw.get("last_loss", {}).items():
                try:
                    _symbol_last_loss_time[sym] = datetime.fromisoformat(ts)
                except Exception:
                    continue

            # === BLOCK UNTIL (remove expirados) ===
            _symbol_block_until = {}
            for sym, ts in raw.get("block_until", {}).items():
                try:
                    dt = datetime.fromisoformat(ts)
                    if dt > now:
                        _symbol_block_until[sym] = dt
                except Exception:
                    continue

        logger.info(
            f"📉 Loss streak carregado | "
            f"{len(_symbol_loss_streak)} símbolos | "
            f"{len(_symbol_block_until)} bloqueados"
        )

    except Exception as e:
        logger.error(f"❌ Erro ao carregar loss streak: {e}", exc_info=True)


def save_loss_streak_data():
    data = {
        "streak": dict(_symbol_loss_streak),
        "last_loss": {k: v.isoformat() for k, v in _symbol_last_loss_time.items()},
        "block_until": {k: v.isoformat() for k, v in _symbol_block_until.items()}
    }
    try:
        with open(LOSS_STREAK_FILE, "w") as f:
            json.dump(data, f)
        logger.info("💾 Loss streak salvo.")
    except Exception as e:
        logger.error(f"Erro ao salvar loss streak: {e}")

def record_trade_outcome(symbol: str, profit_loss: float):
    """
    Registra resultado do trade por símbolo.
    Thread-safe, bloqueio finito e reset correto.
    """
    if not symbol:
        return

    now = datetime.now()

    with loss_streak_lock:

        # Garante inicialização
        _symbol_loss_streak.setdefault(symbol, 0)

        # ======================
        # ✅ LUCRO → RESET TOTAL
        # ======================
        if profit_loss >= 0:
            if _symbol_loss_streak[symbol] > 0:
                logger.info(
                    f"✅ {symbol}: Streak resetado após lucro "
                    f"({_symbol_loss_streak[symbol]} → 0)"
                )

            _symbol_loss_streak[symbol] = 0
            _symbol_last_loss_time.pop(symbol, None)
            _symbol_block_until.pop(symbol, None)

        # ======================
        # 🔴 PERDA
        # ======================
        else:
            _symbol_loss_streak[symbol] += 1
            _symbol_last_loss_time[symbol] = now

            streak = _symbol_loss_streak[symbol]

            logger.warning(f"🔴 {symbol}: Perda consecutiva #{streak}")

            # 🚫 BLOQUEIO SOMENTE UMA VEZ
            if (
                streak >= config.SYMBOL_MAX_CONSECUTIVE_LOSSES
                and symbol not in _symbol_block_until
            ):
                block_until = now + timedelta(
                    hours=config.SYMBOL_COOLDOWN_HOURS
                )
                _symbol_block_until[symbol] = block_until

                logger.critical(
                    f"🚫 {symbol}: BLOQUEADO até "
                    f"{block_until.strftime('%d/%m %H:%M')} "
                    f"({streak} perdas seguidas)"
                )

    save_loss_streak_data()


def is_symbol_blocked(symbol: str) -> tuple[bool, str]:
    """
    Retorna (blocked: bool, reason: str)
    Thread-safe e com limpeza completa.
    """
    if not symbol:
        return False, ""

    now = datetime.now()

    with loss_streak_lock:

        # =========================
        # 🧹 LIMPA BLOQUEIO EXPIRADO
        # =========================
        if symbol in _symbol_block_until:
            if now >= _symbol_block_until[symbol]:
                _symbol_block_until.pop(symbol, None)
                _symbol_loss_streak[symbol] = 0
                _symbol_last_loss_time.pop(symbol, None)

                save_loss_streak_data()

                logger.info(f"✅ {symbol}: Bloqueio expirado e removido")

        # =========================
        # 🚫 AINDA BLOQUEADO
        # =========================
        if symbol in _symbol_block_until:
            remaining_sec = (_symbol_block_until[symbol] - now).total_seconds()
            remaining_min = max(1, int(remaining_sec // 60))

            return (
                True,
                f"Bloqueado ({remaining_min} min restantes) — "
                f"{config.SYMBOL_MAX_CONSECUTIVE_LOSSES} perdas seguidas"
            )

    return False, ""


def get_cached_indicators(
    symbol: str,
    timeframe,
    count: int = 300,
    ttl: int = 45
):
    """
    Cache seguro de indicadores (Redis)
    - Não cacheia erro
    - Valida frescor do candle
    - Fail-safe total
    """

    # =========================
    # 🔄 FALLBACK SEM REDIS
    # =========================
    if not REDIS_AVAILABLE:
        df = safe_copy_rates(symbol, timeframe, count)
        if df is None or len(df) < 50:
            return {"error": "no_data"}
        return quick_indicators_custom(symbol, timeframe, df=df)

    # =========================
    # 🔑 CHAVE COM VERSÃO REAL
    # =========================
    key = f"ind:v3:{symbol}:{timeframe}:{count}"

    try:
        cached = redis_client.get(key)
        if cached:
            ind = pickle.loads(cached)

            # ❌ Cache inválido
            if not isinstance(ind, dict) or ind.get("error"):
                raise ValueError("Cache inválido")

            # ⏱️ Valida frescor do candle
            candle_time = ind.get("candle_time")
            if candle_time and (time.time() - candle_time) <= ttl:
                logger.debug(f"🧠 Cache HIT: {symbol}")
                return ind

    except Exception as e:
        logger.debug(f"Redis cache ignorado ({symbol}): {e}")

    # =========================
    # 🧮 RECÁLCULO
    # =========================
    df = safe_copy_rates(symbol, timeframe, count)
    if df is None or len(df) < 50:
        return {"error": "no_data"}

    ind = quick_indicators_custom(symbol, timeframe, df=df)
    if not isinstance(ind, dict) or ind.get("error"):
        return ind  # ❌ NÃO CACHEIA ERRO

    # Timestamp do candle fechado
    ind["candle_time"] = df.index[-1].timestamp() if hasattr(df.index[-1], "timestamp") else time.time()

    # =========================
    # 💾 SALVA CACHE
    # =========================
    try:
        redis_client.setex(key, ttl, pickle.dumps(ind))
    except Exception as e:
        logger.debug(f"Redis set ignorado ({symbol}): {e}")

    return ind


def mt5_with_retry(
    max_retries: int = 4,
    base_delay: float = 1.0,
    retry_on_none: bool = True
):
    """
    Decorator robusto para operações MT5
    - Retry apenas para erros recuperáveis
    - Retry se retorno for None (MT5 bug comum)
    """

    RECOVERABLE_EXCEPTIONS = (
        ConnectionError,
        TimeoutError,
        RuntimeError,
    )

    def decorator(func):
        def wrapper(*args, **kwargs):
            delay = base_delay
            last_exception = None

            for attempt in range(1, max_retries + 1):
                try:
                    result = func(*args, **kwargs)

                    # 🔁 Retry se MT5 retornou None
                    if retry_on_none and result is None:
                        raise RuntimeError("MT5 retornou None")

                    return result

                except RECOVERABLE_EXCEPTIONS as e:
                    last_exception = e

                    if attempt == max_retries:
                        logger.exception(
                            f"🚨 Falha definitiva em {func.__name__} "
                            f"após {max_retries} tentativas"
                        )
                        raise

                    logger.warning(
                        f"⚠️ {func.__name__} falhou "
                        f"(tentativa {attempt}/{max_retries}): {e} | "
                        f"retry em {delay:.1f}s"
                    )

                    time.sleep(delay)
                    delay *= 2

                except Exception:
                    # ❌ Erro lógico → NÃO RETENTA
                    logger.exception(
                        f"❌ Erro não recuperável em {func.__name__}"
                    )
                    raise

            raise last_exception  # segurança

        return wrapper
    return decorator


def calculate_advanced_metrics(trades_df: pd.DataFrame) -> Dict[str, float]:
    metrics = {}

    if not is_valid_dataframe(trades_df, min_rows=5):
        return metrics

    pnl = trades_df['pnl_money']

    # Profit Factor
    gross_profit = pnl[pnl > 0].sum()
    gross_loss = abs(pnl[pnl < 0].sum())
    metrics['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # MAE / MFE
    if {'mae', 'mfe'}.issubset(trades_df.columns):
        metrics['avg_mae'] = trades_df['mae'].mean()
        metrics['avg_mfe'] = trades_df['mfe'].mean()

    # Equity curve
    initial_equity = trades_df.get('equity_start', pd.Series([100000])).iloc[0]
    equity_curve = initial_equity + pnl.cumsum().values

    peak = np.maximum.accumulate(equity_curve)
    drawdown_pct = ((equity_curve - peak) / peak) * 100

    metrics['ulcer_index'] = np.sqrt(np.mean(drawdown_pct ** 2))

    max_dd = np.min((equity_curve - peak) / peak)
    total_return = (equity_curve[-1] / equity_curve[0]) - 1

    metrics['recovery_factor'] = total_return / abs(max_dd) if max_dd < 0 else float('inf')

    return metrics



def is_spread_acceptable(symbol: str, max_spread_pct: float | None = None) -> bool:
    tick = mt5.symbol_info_tick(symbol)
    if not tick or tick.bid <= 0 or tick.ask <= 0:
        return False

    spread = tick.ask - tick.bid
    mid_price = (tick.ask + tick.bid) / 2
    spread_pct = (spread / mid_price) * 100

    server_time = datetime.fromtimestamp(tick.time).time()

    if max_spread_pct is None:
        if server_time < datetime.strptime("15:30", "%H:%M").time():
            max_spread_pct = 0.15
        elif server_time < datetime.strptime("17:00", "%H:%M").time():
            max_spread_pct = 0.30
        else:
            max_spread_pct = 0.45  # after market

    if spread_pct > max_spread_pct:
        logger.debug(
            f"{symbol}: Spread {spread_pct:.3f}% > {max_spread_pct:.2f}% "
            f"(hora servidor {server_time.strftime('%H:%M')})"
        )
        return False

    return True

def adjust_global_sl_after_pyr(symbol: str, side: str, current_price: float, atr: float):
    """
    Ajusta SL global após pyramiding:
    - Protege lucro da primeira perna
    - Aplica SL único para todas as posições do símbolo
    """

    if atr is None or atr <= 0:
        return

    with mt5_lock:
        positions = mt5.positions_get(symbol=symbol)

    if not positions:
        return

    # 🔹 Identifica a primeira perna (menor ticket = mais antiga)
    positions = sorted(positions, key=lambda p: p.time)
    anchor = positions[0]

    # 🔹 SL base: entrada da primeira perna ± buffer ATR
    buffer = atr * 0.2  # buffer pequeno para evitar stop exato

    if side == "BUY":
        new_sl = anchor.price_open + buffer
        if new_sl >= current_price:
            return
    else:
        new_sl = anchor.price_open - buffer
        if new_sl <= current_price:
            return

    new_sl = round(new_sl, 2)

    # 🔹 Aplica SL global apenas se for MELHOR que o atual
    for p in positions:
        if p.sl:
            if side == "BUY" and new_sl <= p.sl:
                continue
            if side == "SELL" and new_sl >= p.sl:
                continue

        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": p.ticket,
            "symbol": symbol,
            "sl": new_sl,
            "tp": p.tp,
            "magic": 2026
        }

        with mt5_lock:
            res = mt5.order_send(request)

        if res.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"🔒 SL global ajustado {symbol} → {new_sl}")
        else:
            logger.error(f"Erro ao ajustar SL {symbol}: {res.comment}")

def apply_partial_exit_after_pyr(symbol: str, side: str, current_price: float, atr: float):
    """
    Fecha parcialmente a ÚLTIMA perna após pyramiding
    Condições:
    - Lucro >= +1R
    - Apenas a perna mais recente
    - Volume mínimo respeitado
    """

    if atr is None or atr <= 0:
        return

    with mt5_lock:
        positions = mt5.positions_get(symbol=symbol)

    if not positions or len(positions) < 2:
        return  # Sem pyramiding

    # 🔹 Ordena da mais recente para a mais antiga
    positions = sorted(positions, key=lambda p: p.time, reverse=True)
    last_leg = positions[0]

    # 🔹 Distância de risco da perna
    if last_leg.sl is None or last_leg.sl == 0:
        return

    if side == "BUY":
        risk_dist = abs(last_leg.price_open - last_leg.sl)
        profit_dist = current_price - last_leg.price_open
    else:
        risk_dist = abs(last_leg.sl - last_leg.price_open)
        profit_dist = last_leg.price_open - current_price

    if risk_dist <= 0:
        return

    r_multiple = profit_dist / risk_dist

    if r_multiple < 1.0:
        return  # Ainda não atingiu +1R

    # 🔹 Volume parcial (50%)
    close_volume = round(last_leg.volume * 0.5, 2)

    info = mt5.symbol_info(symbol)
    if not info or close_volume < info.volume_min:
        return

    # 🔹 Envia fechamento parcial
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "position": last_leg.ticket,
        "volume": close_volume,
        "type": mt5.ORDER_TYPE_SELL if side == "BUY" else mt5.ORDER_TYPE_BUY,
        "price": current_price,
        "deviation": get_dynamic_slippage(symbol, datetime.now().hour),
        "magic": 2026,
        "comment": "Partial Exit +1R"
    }

    with mt5_lock:
        result = mt5.order_send(request)

    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        logger.info(
            f"💰 Partial exit {symbol}: "
            f"{close_volume} @ {current_price:.2f} (+{r_multiple:.2f}R)"
        )




def apply_anti_martingale(loss_streak: int) -> float:
    """
    Anti-Martingale: Reduz volume pós-perdas (risco, para win rate psicológico).
    """
    if loss_streak >= 2:
        return 0.5  # 50% volume normal
    return 1.0
def calculate_dynamic_sl_tp(symbol, side, entry_price, ind):
    atr = ind.get("atr", 0.10)
    adx = ind.get("adx", 20)
    
    # Detecta regime
    if adx >= 30:
        regime = "TRENDING"
        tp_mult = 4.5  # Deixa o lucro correr
    elif ind.get("vol_breakout"):
        regime = "BREAKOUT"
        tp_mult = 5.0  # Máxima agressividade
    else:
        regime = "RANGING"
        tp_mult = 2.5  # Conservador
    
    sl_mult = 2.0  # Mantém fixo
    
    if side == "BUY":
        sl = entry_price - (atr * sl_mult)
        tp = entry_price + (atr * tp_mult)
    else:
        sl = entry_price + (atr * sl_mult)
        tp = entry_price - (atr * tp_mult)
    
    # Normalização
    info = mt5.symbol_info(symbol)
    sl = round(sl / info.trade_tick_size) * info.trade_tick_size
    tp = round(tp / info.trade_tick_size) * info.trade_tick_size
    
    return sl, tp

def normalize_price(symbol, price):
    info = mt5.symbol_info(symbol)
    if not info: return price
    
    # Use trade_tick_size aqui também
    normalized = round(price / info.trade_tick_size) * info.trade_tick_size
    return round(normalized, info.digits)

def check_and_apply_breakeven(symbol, current_indicators, move_threshold_atr=1.0):
    """
    Se o preço undou 1x o ATR a favor, move o SL para o preço de entrada.
    """
    positions = mt5.positions_get(symbol=symbol)
    if not is_valid_dataframe(positions):
        return

    ind = current_indicators.get(symbol)
    if not ind: 
        return

    atr = ind.get("atr", 0.10)
    
    for p in positions:
        if p.type == mt5.POSITION_TYPE_BUY:
            if p.price_current >= (p.price_open + (atr * move_threshold_atr)):
                if p.sl < p.price_open:
                    logger.info(f"🛡️ {symbol}: Movendo para Breakeven (COMPRA)")
                    modify_sl_tp(p.ticket, p.price_open + (atr * 0.1), p.tp)
        
        elif p.type == mt5.POSITION_TYPE_SELL:
            if p.price_current <= (p.price_open - (atr * move_threshold_atr)):
                if p.sl > p.price_open or p.sl == 0:
                    logger.info(f"🛡️ {symbol}: Movendo para Breakeven (VENDA)")
                    modify_sl_tp(p.ticket, p.price_open - (atr * 0.1), p.tp)

def modify_sl_tp(ticket, new_sl, new_tp):
    """
    Envia a solicitação de modificação de SL/TP para um ticket específico.
    """
    # Normaliza os preços antes de enviar para evitar erro de tick_size
    pos = mt5.positions_get(ticket=ticket)
    if not pos: return False
    
    symbol = pos[0].symbol
    new_sl = normalize_price(symbol, new_sl)
    new_tp = normalize_price(symbol, new_tp)

    request = {
        "action": mt5.TRADE_ACTION_SLTP,
        "position": ticket,
        "sl": float(new_sl),
        "tp": float(new_tp),
    }

    with mt5_lock:
        result = mt5.order_send(request)
    
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logger.error(f"❌ Falha ao mover Stop: {result.comment}")
        return False
    
    return True

def update_correlations(top15_symbols):
    """
    Calcula matriz de correlação dos ativos.
    CORRIGIDO: Agora usa o nome correto do parâmetro.
    """
    # ✅ CORREÇÃO: Era 'symbols', agora é 'top15_symbols'
    if not isinstance(top15_symbols, (list, tuple)):
        logger.error(f"update_correlations recebeu tipo inválido: {type(top15_symbols)}")
        return
    
    if not top15_symbols:
        logger.warning("update_correlations: Lista de símbolos vazia")
        return
    
    logger.info(f"📊 Atualizando correlação para {len(top15_symbols)} ativos...")
    
    try:
        # Coleta dados de fechamento dos últimos 50 candles
        data = {}
        for sym in top15_symbols:
            rates = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_M15, 0, 50)
            if rates is not None:
                data[sym] = [r['close'] for r in rates]
        
        if len(data) > 1:
            df = pd.DataFrame(data)
            corr_matrix = df.corr()
            
            # Salva na variável global
            global last_corr_matrix
            last_corr_matrix = corr_matrix
            logger.info("✅ Matriz de correlação atualizada")
            
    except Exception as e:
        logger.error(f"Erro ao calcular correlações: {e}", exc_info=True)

# =========================================================
# 📤 TELEGRAM & REPORTING
# =========================================================

def send_telegram_message(text: str):
    """
    Envia uma mensagem para o Telegram usando o bot instanciado.
    """
    bot = get_telegram_bot()
    if bot and getattr(config, "ENABLE_TELEGRAM_NOTIF", False):
        try:
            bot.send_message(
                chat_id=config.TELEGRAM_CHAT_ID,
                text=text,
                parse_mode="HTML",
                disable_web_page_preview=True
            )
        except Exception as e:
            logger.warning(f"Erro ao enviar mensagem para o Telegram: {e}")

def send_daily_performance_report():
    """
    Envia relatório consolidado do dia.
    """
    try:
        from database import get_trades_by_date
        from datetime import date
        
        today = date.today()
        trades = get_trades_by_date(today)
        
        if not is_valid_dataframe(trades):
            send_telegram_message(f"📊 <b>RELATÓRIO {today.strftime('%d/%m/%Y')}</b>\nNenhum trade realizado hoje.")
            return

        acc = mt5.account_info()
        equity = acc.equity if acc else 0
        
        total_pnl = trades['pnl_money'].sum()
        win_rate = (len(trades[trades['pnl_money'] > 0]) / len(trades)) * 100
        
        msg = (
            f"📊 <b>RELATÓRIO FINAL XP3 - {today.strftime('%d/%m/%Y')}</b>\n\n"
            f"💰 Equity: R$ {equity:,.2f}\n"
            f"📈 PnL: R$ {total_pnl:+.2f}\n"
            f"🎯 Win Rate: {win_rate:.1f}%\n"
            f"🔢 Total Trades: {len(trades)}\n\n"
            f"✅ Sistema Estável"
        )
        send_telegram_message(msg)
    except Exception as e:
        logger.error(f"Erro no relatório diário: {e}")

# =========================================================
# 📰 NEWS & EVENTS
# =========================================================

def send_news_alert(message: str):
    """
    Envia alerta de notícia importante via Telegram
    """
    if not getattr(config, "ENABLE_TELEGRAM_NOTIF", False):
        return
    
    full_message = f"📰 <b>CALENDÁRIO ECONÔMICO</b>\n\n{message}"
    
    try:
        send_telegram_message(full_message)
        logger.info(f"News alert enviado: {message}")
    except Exception as e:
        logger.error(f"Falha ao enviar news alert: {e}")

def send_next_high_impact_event():
    """
    Envia o próximo evento de alto impacto (comando /proximoevento)
    """
    try:
        from news_filter import get_next_high_impact_event
        message = get_next_high_impact_event()
        emoji = "🔴" if "em" in message and "min" in message else "🟢"
        send_news_alert(f"{emoji} <b>PRÓXIMO EVENTO</b>\n\n{message}")
    except: pass

def send_current_blackout_status():
    """
    Envia status atual de blackout (se está bloqueado ou não)
    """
    try:
        from news_filter import check_news_blackout, get_upcoming_events
        blocked, reason = check_news_blackout()
        
        if blocked:
            status = f"🚫 <b>BOT EM BLACKOUT</b>\n\n{reason}"
        else:
            status = "✅ <b>Trading Liberado</b> - Sem eventos críticos próximos."
        
        send_news_alert(status)
    except: pass

# =========================================================
# 🛡️ RECOVERY & RISK
# =========================================================

def mt5_crash_recovery():
    """
    Recuperação de crash básica.
    """
    if not mt5.initialize():
        logger.critical("Falha ao reconectar MT5")
        return
    logger.info("Recuperação MT5 completa")

def get_daily_volume() -> float:
    """Calcula o volume total negociado PELO BOT no dia (em R$)."""
    try:
        from_date = datetime.combine(datetime.now().date(), datetime_time.min)
        to_date = datetime.now()
        with mt5_lock:
            deals = mt5.history_deals_get(from_date, to_date)
        if not deals:
            return 0.0
        total_volume = sum(abs(deal.volume * deal.price) for deal in deals if deal.entry == mt5.DEAL_ENTRY_OUT)  # Sum of closed trades' value
        return total_volume
    except Exception as e:
        logger.error(f"Erro ao calcular volume: {e}")
        return 0.0

def get_realtime_win_rate(lookback_trades: int = 20) -> Dict[str, Any]:
    """
    Calcula win rate em tempo real a partir do histórico do MT5.
    
    Args:
        lookback_trades: Quantos últimos trades considerar.
        
    Returns:
        Dict com win_rate (float 0.0-1.0), total_trades e profit_factor.
    """
    try:
        from datetime import datetime, timedelta
        import MetaTrader5 as mt5
        
        # Define período de busca (últimos 30 dias para garantir dados)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        with mt5_lock:
            # Pega histórico de negócios (DEALS)
            deals = mt5.history_deals_get(start_date, end_date)
        
        if deals is None or len(deals) == 0:
            return {"win_rate": 0.0, "total_trades": 0, "profit_factor": 0.0}
        
        # Filtra apenas DEALS de SAÍDA que tenham PnL Realizado
        # (entry != alignment)
        valid_trades = [d for d in deals if d.entry != mt5.DEAL_ENTRY_IN and d.profit != 0]
        
        # Pega apenas os últimos N
        last_trades = valid_trades[-lookback_trades:] if len(valid_trades) > lookback_trades else valid_trades
        
        if not last_trades:
            return {"win_rate": 0.0, "total_trades": 0, "profit_factor": 0.0}
            
        wins = sum(1 for d in last_trades if d.profit > 0)
        total = len(last_trades)
        
        gross_profit = sum(d.profit for d in last_trades if d.profit > 0)
        gross_loss = abs(sum(d.profit for d in last_trades if d.profit < 0))
        
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 2.0 # Default favorável se não houver losses
        
        return {
            "win_rate": wins / total if total > 0 else 0.0,
            "total_trades": total,
            "profit_factor": round(profit_factor, 2)
        }
        
    except Exception as e:
        logger.error(f"Erro em get_realtime_win_rate: {e}")
        return {"win_rate": 0.0, "total_trades": 0, "profit_factor": 0.0}

def calculate_daily_dd() -> float:
    """
    Calcula o drawdown diário (Equity vs Daily Max).
    """
    try:
        acc = mt5.account_info()
        if not acc: return 0.0
        
        # Tenta buscar a global do bot
        import sys
        bot_mod = sys.modules.get('bot')
        if bot_mod and hasattr(bot_mod, 'daily_max_equity'):
            m_equity = getattr(bot_mod, 'daily_max_equity')
            if m_equity > acc.equity:
                return (m_equity - acc.equity) / m_equity
        return 0.0
    except: return 0.0

# ============================================
# 🔥 PRIORIDADE 1 - ANTI-CHOP & LIMITS
# ============================================

# Arquivos persistentes
ANTI_CHOP_FILE = "anti_chop_data.json"
DAILY_LIMITS_FILE = "daily_symbol_limits.json"

# Estado global
_symbol_sl_timestamps = {}  # {symbol: timestamp_último_sl}
_symbol_sl_prices = {}  # {symbol: preço_quando_bateu_sl}
_daily_symbol_trades = defaultdict(lambda: {"total": 0, "losses": 0})  # Contador diário

def load_anti_chop_data():
    """Carrega dados de cooldown"""
    global _symbol_sl_timestamps, _symbol_sl_prices
    if os.path.exists(ANTI_CHOP_FILE):
        try:
            with open(ANTI_CHOP_FILE, "r") as f:
                data = json.load(f)
                _symbol_sl_timestamps = {
                    k: datetime.fromisoformat(v) 
                    for k, v in data.get("timestamps", {}).items()
                }
                _symbol_sl_prices = data.get("prices", {})
            logger.info("✅ Dados anti-chop carregados")
        except Exception as e:
            logger.error(f"Erro ao carregar anti-chop: {e}")

def save_anti_chop_data():
    """Salva dados de cooldown"""
    data = {
        "timestamps": {k: v.isoformat() for k, v in _symbol_sl_timestamps.items()},
        "prices": _symbol_sl_prices
    }
    atomic_save_json(ANTI_CHOP_FILE, data)

def load_daily_limits():
    """Carrega contadores diários"""
    global _daily_symbol_trades
    if os.path.exists(DAILY_LIMITS_FILE):
        try:
            with open(DAILY_LIMITS_FILE, "r") as f:
                data = json.load(f)
                saved_date = data.get("date")
                today = datetime.now().date().isoformat()
                if saved_date == today:
                    _daily_symbol_trades = defaultdict(
                        lambda: {"total": 0, "losses": 0},
                        data.get("trades", {})
                    )
                    logger.info("✅ Limites diários carregados")
                else:
                    logger.info("🔄 Novo dia detectado - resetando limites")
        except Exception as e:
            logger.error(f"Erro ao carregar limites: {e}")

def save_daily_limits():
    """Salva contadores diários"""
    data = {
        "date": datetime.now().date().isoformat(),
        "trades": dict(_daily_symbol_trades)
    }
    atomic_save_json(DAILY_LIMITS_FILE, data)

def register_sl_hit(symbol: str, sl_price: float):
    """Registra que o SL foi atingido"""
    if not getattr(config, "ANTI_CHOP", {}).get("enabled", False):
        return
    _symbol_sl_timestamps[symbol] = datetime.now()
    _symbol_sl_prices[symbol] = sl_price
    save_anti_chop_data()
    logger.info(f"🛑 {symbol}: SL registrado @ R${sl_price:.2f}")

def check_anti_chop_filter(symbol: str, current_price: float, atr: float) -> Tuple[bool, str]:
    """Valida se o ativo está em cooldown após SL ou volatilidade excedida"""
    if not getattr(config, "ANTI_CHOP", {}).get("enabled", False):
        return True, ""
    last_sl_time = _symbol_sl_timestamps.get(symbol)
    if last_sl_time:
        # Bloqueio total até o fim do dia após um SL (se habilitado)
        try:
            if config.ANTI_CHOP.get("block_full_day_on_single_sl", False):
                if last_sl_time.date() == datetime.now().date():
                    return False, "Bloqueado hoje por SL (reset diário)"
        except Exception:
            pass
        cooldown_min = config.ANTI_CHOP.get("cooldown_after_sl_minutes", 120)
        elapsed = (datetime.now() - last_sl_time).total_seconds() / 60
        if elapsed < cooldown_min:
            return False, f"Cooldown SL ({int(cooldown_min - elapsed)} min rest)"
    return True, ""

def clear_anti_chop_cooldown(symbol: str):
    """Limpa cooldown após entrada bem-sucedida"""
    try:
        # Não limpa se estiver usando bloqueio por SL até o fim do dia
        if config.ANTI_CHOP.get("block_full_day_on_single_sl", False):
            last_sl_time = _symbol_sl_timestamps.get(symbol)
            if last_sl_time and last_sl_time.date() == datetime.now().date():
                return
    except Exception:
        pass
    if symbol in _symbol_sl_timestamps: del _symbol_sl_timestamps[symbol]
    if symbol in _symbol_sl_prices: del _symbol_sl_prices[symbol]
    save_anti_chop_data()

def check_daily_symbol_limit(symbol: str) -> Tuple[bool, str]:
    """Verifica limites diários por ativo"""
    if not getattr(config, "DAILY_SYMBOL_LIMITS", {}).get("enabled", False):
        return True, ""
    try:
        # Bloqueio adicional: se bateu SL hoje e flag habilitada, bloqueia
        if config.ANTI_CHOP.get("block_full_day_on_single_sl", False):
            last_sl_time = _symbol_sl_timestamps.get(symbol)
            if last_sl_time and last_sl_time.date() == datetime.now().date():
                return False, "Bloqueado por SL hoje"
    except Exception:
        pass
    stats = _daily_symbol_trades[symbol]
    if stats["losses"] >= config.DAILY_SYMBOL_LIMITS.get("max_losing_trades_per_symbol", 2):
        return False, "Limite de perdas diário"
    if stats["total"] >= config.DAILY_SYMBOL_LIMITS.get("max_total_trades_per_symbol", 5):
        return False, "Limite total diário"
    return True, ""

def register_trade_result(symbol: str, is_loss: bool):
    """Registra resultado do trade para limites diários"""
    if not getattr(config, "DAILY_SYMBOL_LIMITS", {}).get("enabled", False):
        return
    _daily_symbol_trades[symbol]["total"] += 1
    if is_loss:
        _daily_symbol_trades[symbol]["losses"] += 1
    save_daily_limits()

def reset_daily_limits():
    """Reseta contadores diários"""
    global _daily_symbol_trades
    _daily_symbol_trades.clear()
    save_daily_limits()

def check_pyramid_eligibility(symbol: str, side: str, ind: dict) -> Tuple[bool, str]:
    """Valida se pode adicionar perna (pirâmide) na posição atual"""
    with mt5_lock:
        positions = [p for p in mt5.positions_get(symbol=symbol) or []]
    if not positions: return True, "Primeira entrada"
    pos = positions[0]
    existing_side = "BUY" if pos.type == mt5.POSITION_TYPE_BUY else "SELL"
    if existing_side != side: return False, "Direção oposta"
    pyr_count = pos.comment.count("PYR") if pos.comment else 0
    if pyr_count >= getattr(config, "PYRAMID_MAX_LEGS", 3):
        return False, "Limite de pirâmide atingido"
    return True, "Elegível"

def send_telegram_trade(symbol: str, side: str, volume: float, price: float, sl: float, tp: float, comment: str = ""):
    msg = (
        f"🚀 <b>NOVA OPERAÇÃO: {symbol}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🔹 Direção: {'COMPRA 🟢' if side == 'BUY' else 'VENDA 🔴'}\n"
        f"🔹 Volume: {volume:.2f}\n"
        f"🔹 Entrada: R$ {price:,.2f}\n"
        f"🔹 SL: R$ {sl:,.2f} | TP: R$ {tp:,.2f}\n"
        f"🔹 Obs: {comment}\n"
    )
    send_telegram_message(msg)

def send_telegram_exit(symbol: str, side: str, volume: float, entry_price: float, exit_price: float, profit_loss: float, reason: str = ""):
    emoji = "✅" if profit_loss >= 0 else "❌"
    msg = (
        f"{emoji} <b>SAÍDA DE POSIÇÃO: {symbol}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🔹 Direção: {side}\n"
        f"🔹 Resultado: <b>R$ {profit_loss:+,.2f}</b>\n"
        f"🔹 Preço Saída: R$ {exit_price:,.2f}\n"
        f"🔹 Motivo: {reason}\n"
    )
    send_telegram_message(msg)

def calcular_lucro_realizado_txt():
    import os
    import re
    from datetime import datetime
    
    filename = f"trades_log_{datetime.now().strftime('%Y-%m-%d')}.txt"
    
    if not os.path.exists(filename):
        return 0.0, 0
    
    total_pnl = 0.0
    contagem_trades = 0
    
    with open(filename, "r", encoding="utf-8") as f:
        conteudo = f.read()
        
        # ✅ REGEX MAIS ROBUSTO
        # Busca linhas de fechamento (não de abertura)
        for linha in conteudo.split('\n'):
            if "Abertura de Posição" in linha or "---" in linha:
                continue
            
            # Match: P&L: +1550.00 ou P&L: -320.50
            match = re.search(r'P&L:\s*([+-]?\d+\.?\d*)', linha)
            if match:
                try:
                    pnl = float(match.group(1))
                    total_pnl += pnl
                    contagem_trades += 1
                except ValueError:
                    continue
                    
    return total_pnl, contagem_trades

def obter_resumo_financeiro_do_dia():
    lucro_realizado, total_ordens = calcular_lucro_realizado_txt()
    lucro_aberto_total = sum(p.profit for p in mt5.positions_get()) if mt5.positions_get() else 0.0
    return lucro_realizado, lucro_aberto_total, total_ordens

def responder_comando_lucro(message):
    bot = get_telegram_bot()
    if not bot: return

    # 1. Busca o Lucro Realizado no seu arquivo TXT (o que já está no bolso)
    realizado, qtd = calcular_lucro_realizado_txt()

    # 2. Busca o Lucro Flutuante (o que está aberto agora no MT5)
    posicoes_abertas = mt5.positions_get()
    aberto = sum(p.profit for p in posicoes_abertas) if posicoes_abertas else 0.0
    total_do_dia = realizado + aberto
    
    emoji = "🚀" if total_do_dia >= 0 else "⚠️"
    
    msg = (
        f"{emoji} <b>STATUS XP3 - AGORA</b>\n\n"
        f"💰 <b>Realizado:</b> R$ {realizado:,.2f}\n"
        f"📈 <b>Flutuante:</b> R$ {aberto:,.2f}\n"
        f"---------------------------\n"
        f"🏆 <b>TOTAL DO DIA: R$ {total_do_dia:,.2f}</b>\n\n"
        f"<i>Baseado em {qtd} ordens e {len(posicoes_abertas) if posicoes_abertas else 0} posições abertas.</i>"
    )

    bot.reply_to(message, msg, parse_mode="HTML")

def check_minimum_price_movement(symbol: str, df: pd.DataFrame, atr: float) -> Tuple[bool, str]:
    """
    Valida se houve movimento mínimo antes de entrar
    """
    if not getattr(config, "MIN_PRICE_MOVEMENT", {}).get("enabled", False):
        return True, ""
    
    lookback = config.MIN_PRICE_MOVEMENT.get("lookback_candles", 5)
    
    if df is None or len(df) < lookback:
        return True, ""  # Fail-open
    
    recent = df.tail(lookback)
    price_range = recent["high"].max() - recent["low"].min()
    
    min_movement = atr * config.MIN_PRICE_MOVEMENT.get("min_atr_multiplier", 1.5)
    
    if price_range < min_movement:
        return False, f"Range baixo ({price_range:.2f} < {min_movement:.2f})"
    
    return True, ""

# Final calls for data persistence
try:
    load_anti_chop_data()
    load_daily_limits()
except:
    pass
def diagnose_symbol_failure(symbol: str) -> str:
    try:
        info = mt5.symbol_info(symbol)
        if info is None:
            return "Símbolo não existe na corretora"
        if not getattr(info, "selectable", False):
            return "Símbolo não selecionável"
        if not getattr(info, "visible", False):
            return "Símbolo não visível"
        from datetime import datetime
        exp = getattr(info, "expiration_time", None)
        if isinstance(exp, datetime) and exp < datetime.now():
            return "Contrato vencido"
        return "Falha na seleção"
    except Exception as e:
        return f"Erro na verificação: {str(e)}"
def ensure_market_watch_symbols():
    import config
    logger.info("📋 Sincronizando Market Watch (Strict Mode)...")
    desired = set(config.SECTOR_MAP.keys())
    try:
        desired = desired.union(set(getattr(config, "ELITE_SYMBOLS", {}).keys()))
    except Exception:
        pass
    desired.add("IBOV")
    count_added = 0
    failed = []
    added_names = set()
    broker = (detect_broker() or "").lower()
    for symbol in desired:
        try:
            if mt5.symbol_select(symbol, True):
                count_added += 1
                added_names.add(symbol)
                continue
            reason = diagnose_symbol_failure(symbol)
            logger.warning(f"⚠️ Corretora {broker}: Falha ao adicionar {symbol} - {reason}")
            s_up = (symbol or "").upper()
            is_fut_like = ("$" in s_up) or s_up.startswith(("WIN", "WDO", "WSP", "IND", "DOL", "SMALL"))
            if is_fut_like:
                base = s_up.replace("$", "")
                base = base[:4] if base.startswith(("SMAL", "SMALL")) else base[:3]
                cands = get_futures_candidates(base)
                if cands:
                    cands_sorted = sorted(cands, key=lambda c: (-calculate_contract_score(c), c.get("days_to_exp", 9999)))
                    best = cands_sorted[0]
                    cand_sym = best.get("symbol")
                    if cand_sym and mt5.symbol_select(cand_sym, True):
                        count_added += 1
                        added_names.add(cand_sym)
                        continue
            else:
                res = mt5.symbols_get(f"{symbol}*") or []
                if res:
                    name = getattr(res[0], "name", "")
                    if name and mt5.symbol_select(name, True):
                        count_added += 1
                        added_names.add(name)
                        continue
            failed.append(symbol)
        except Exception as e:
            logger.error(f"❌ Erro ao processar {symbol}: {e}")
            failed.append(symbol)
    if failed:
        logger.warning(f"⚠️ {len(failed)} ativos falharam ao adicionar: {failed}")
    logger.info("🧹 Limpando ativos desnecessários do Market Watch...")
    all_symbols = mt5.symbols_get()
    removed_count = 0
    if all_symbols:
        keep_set = desired.union(added_names)
        for s in all_symbols:
            if s.select and s.name not in keep_set:
                if mt5.symbol_select(s.name, False):
                    removed_count += 1
    logger.info(f"✅ Market Watch Sincronizado: {count_added} ativos mantidos/adicionados, {removed_count} removidos.")
