# =============================================
#  utils.py – BOT ELITE 2026 (VERSÃO CORRIGIDA)
# =============================================

import logging
import logging.handlers
import json
import time
import os
import sys
import math
import traceback
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import pandas_ta as ta
from colorama import Fore, Style, init
from functools import wraps
from typing import Any, Optional, Tuple, List
import config
from datetime import datetime, timezone

# ============================================================
#                 LOGGER EM FORMATO JSON
# ============================================================

class JsonFormatter(logging.Formatter):
    def format(self, record):
        data = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
            "funcName": record.funcName,
            "lineno": record.lineno,
        }
        if record.exc_info:
            data["exc"] = self.formatException(record.exc_info)
        return json.dumps(data, ensure_ascii=False)


def setup_logger(log_file: str = config.LOG_FILE):
    logger = logging.getLogger("bot_elite")
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        rot = logging.handlers.RotatingFileHandler(log_file, maxBytes=8_000_000, backupCount=4, encoding="utf-8")
        rot.setFormatter(JsonFormatter())

        cons = logging.StreamHandler(sys.stdout)
        cons.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))

        logger.addHandler(rot)
        logger.addHandler(cons)

    return logger


logger = setup_logger()
init(autoreset=True)

VERDE = Fore.GREEN + Style.BRIGHT
VERMELHO = Fore.RED + Style.BRIGHT
AMARELO = Fore.YELLOW + Style.BRIGHT
AZUL = Fore.CYAN + Style.BRIGHT
ROXO = Fore.MAGENTA + Style.BRIGHT
RESET = Style.RESET_ALL


# ============================================================
#                DECORADOR DE RETENTATIVA
# ============================================================

def retry_on_exception(max_retries=3, delay=0.3):
    def deco(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception:
                    logger.exception(f"Erro em {func.__name__} | tentativa {attempt}/{max_retries}")
                    if attempt == max_retries:
                        raise
                    time.sleep(delay)
        return wrapper
    return deco


# ============================================================
#      SAFE COPY RATES (ROBUSTO + NORMALIZAÇÃO)
# ============================================================

@retry_on_exception(max_retries=2)
def safe_copy_rates(symbol: str, timeframe: int, start: int = 0, count: int = 200):
    try:
        data = mt5.copy_rates_from_pos(symbol, timeframe, start, count)
        if data is None:
            return None

        df = pd.DataFrame(data)
        df.columns = [c.lower() for c in df.columns]
        return df.to_dict("records")
    except Exception:
        logger.exception(f"safe_copy_rates falhou para {symbol}")
        return None


# ============================================================
#       FUNÇÃO UNIVERSAL: EXTRAI ESCALAR DE QUALQUER LIXO
# ============================================================

def _scalar(x):
    """Extrai float de células contendo listas/tuplas/strings/etc."""
    try:
        if isinstance(x, (list, tuple, set)):
            return _scalar(next(iter(x)))
        if isinstance(x, dict):
            return _scalar(next(iter(x.values())))
        if isinstance(x, str):
            x = x.replace(",", ".")
            return float(x)
        return float(x)
    except:
        return np.nan


# ============================================================
#              PREPARAÇÃO DE DADOS (CORRIGIDO)
# ============================================================

def prepare_data_for_scan(symbol: str, params: dict, lookback_days: int = 120):
    try:
        raw = safe_copy_rates(symbol, config.TIMEFRAME_MT5, 0, lookback_days * 48)
        if raw is None or len(raw) < 50:
            return None

        df = pd.DataFrame(raw)
        df.columns = [c.lower() for c in df.columns]

        needed = ["open", "high", "low", "close", "tick_volume"]
        for c in needed:
            if c not in df.columns:
                logger.error(f"{symbol} → coluna ausente: {c}")
                return None

        # Converte tudo para escalar
        for col in ["open", "high", "low", "close", "tick_volume"]:
            df[col] = df[col].apply(_scalar)

        # volume real
        if "real_volume" in df.columns:
            df["volume"] = df["real_volume"].apply(_scalar)
        else:
            df["volume"] = df["tick_volume"]

        df = df.replace([np.inf, -np.inf], np.nan).dropna()

        if df.empty:
            return None

        # ATR básico para stops
        tr = np.maximum(df["high"] - df["low"], np.abs(df["high"] - df["close"].shift(1)))
        atr14 = tr.rolling(14).mean().iloc[-1]

        return df, float(atr14)

    except Exception:
        logger.exception(f"prepare_data_for_scan falhou para {symbol}")
        return None


# ============================================================
#              INDICADORES + ANÁLISE DO ATIVO
# ============================================================

def analyze_symbol_for_trade(symbol: str, timeframe: int, params: dict) -> dict:
    try:
        prep = prepare_data_for_scan(symbol, params)
        if not prep:
            return {"symbol": symbol, "side": params.get("side", "COMPRA"),
                    "score": 0.0, "details": {"STATUS": "SEM_DADOS"}}

        df, atr = prep
        side = params.get("side", "COMPRA")

        # =============== INDICADORES ===============
        df["EMA_FAST"] = ta.ema(df["close"], length=params.get("ema_fast_period", 5))
        df["EMA_SLOW"] = ta.ema(df["close"], length=params.get("ema_slow_period", 25))

        df["RSI_14"] = ta.rsi(df["close"], length=14)

        adx = ta.adx(df["high"], df["low"], df["close"], length=14)
        df["ADX_14"] = adx["ADX_14"]

        df["MOMENTUM_PCT"] = df["close"].pct_change(params.get("momentum_period", 5)) * 100
        df["VOL_MED_20"] = df["volume"].rolling(20).mean()

        last = df.dropna().iloc[-1]

        # =============== SINAL ===============
        sinal, det = check_trade_signal(df, params, side)

        det["CLOSE_PRICE"] = last["close"]
        det["VOLUME_ATUAL"] = last["volume"]

        return {
            "symbol": symbol,
            "side": side,
            "score": det.get("SCORE", 0.0),
            "details": det
        }

    except Exception:
        logger.exception(f"Erro em analyze_symbol_for_trade({symbol})")
        return {"symbol": symbol, "side": params.get("side", "COMPRA"), "score": 0.0,
                "details": {"STATUS": "ERRO_ANALISE"}}


# ============================================================
#              CHECK DE SINAL (CORRIGIDO)
# ============================================================

def check_trade_signal(df: pd.DataFrame, params: dict, side: str):
    try:
        last = df.dropna().iloc[-1]

        ema_fast = last["EMA_FAST"]
        ema_slow = last["EMA_SLOW"]
        rsi = last["RSI_14"]
        adx = last["ADX_14"]
        mom = last["MOMENTUM_PCT"]
        vol = last["volume"]
        vmed = last["VOL_MED_20"]

        det = {
            "EMA_FAST": float(ema_fast),
            "EMA_SLOW": float(ema_slow),
            "RSI": float(rsi),
            "ADX": float(adx),
            "MOMENTUM_%": float(mom),
            "VOL_MED_20": float(vmed),
            "VOLUME_ATUAL": float(vol),
            "PRECO": float(last["close"])
        }

        cond = []

        if side == "COMPRA":
            cond += [
                ema_fast > ema_slow,
                rsi > params.get("rsi_level", 60),
                adx > params.get("adx_min", 25),
                mom > params.get("momentum_min_pct", 0.4),
                vol > vmed * 0.8
            ]
        else:
            cond += [
                ema_fast < ema_slow,
                rsi < (100 - params.get("rsi_level", 60)),
                adx > params.get("adx_min", 25),
                mom < -params.get("momentum_min_pct", 0.4),
                vol > vmed * 0.8
            ]

        if all(cond):
            det["STATUS"] = "SINAL_FORTE"
            det["SCORE"] = 10.0
        else:
            det["STATUS"] = "REPROVADO"
            det["SCORE"] = 0.0

        return det["SCORE"] > 0, det

    except Exception:
        logger.exception("Erro no check_trade_signal()")
        return False, {"STATUS": "ERRO"}


# ============================================================
#         EXECUÇÃO PARALELA (SCAN)
# ============================================================

def execute_parallel_scan(symbols: List[str], params: dict, cb_active: bool):
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results = []
    with ThreadPoolExecutor(max_workers=8) as exe:
        futures = {exe.submit(analyze_symbol_for_trade, sym, config.TIMEFRAME_MT5, params): sym
                   for sym in symbols}

        for f in as_completed(futures):
            try:
                results.append(f.result())
            except Exception:
                logger.exception(f"Erro analisando {futures[f]}")

    return results

def generate_scanner_top10_elite(results):
    """
    Gera o painel TOP10 dos ativos analisados.
    Compatível com o novo utils/analyze_symbol_for_trade.
    """

    if not results:
        return f"{VERMELHO}Nenhum resultado disponível para exibir no TOP10.{RESET}"

    # ordenar por score
    ordered = sorted(results, key=lambda x: x.get("score", 0), reverse=True)
    top10 = ordered[:10]

    header = (
        "\n" +
        "================================================================================================\n"
        "                  TOP 10 FORÇADO - TODOS OS CRITÉRIOS COLORIDOS\n"
        "================================================================================================\n"
        "POS ATIVO  SCORE  LADO   EMA      RSI      ADX     MOM%    VOL        PREÇO    STATUS\n"
        "────────────────────────────────────────────────────────────────────────────────────────────────\n"
    )

    lines = []

    for idx, item in enumerate(top10, start=1):
        sym = item.get("symbol", "?")
        side = item.get("side", "?")
        d = item.get("details", {})

        # Extrair valores
        score = d.get("SCORE", 0.0)
        ema_fast = d.get("EMA_FAST", 0)
        ema_slow = d.get("EMA_SLOW", 0)
        rsi = d.get("RSI", 0)
        adx = d.get("ADX", 0)
        mom = d.get("MOMENTUM_%", 0)
        vol = d.get("VOLUME_ATUAL", 0)
        vol_med = d.get("VOL_MED_20", 0)
        price = d.get("PRECO", 0)
        status = d.get("STATUS", "SEM_DADOS")

        # Cores
        c_score = VERDE if score > 0 else VERMELHO
        c_ema = VERDE if ema_fast > ema_slow else VERMELHO
        c_rsi = VERDE if rsi > 60 else VERMELHO
        c_adx = VERDE if adx > 25 else VERMELHO
        c_mom = VERDE if mom > 0 else VERMELHO
        c_vol = VERDE if vol > vol_med else VERMELHO
        c_status = VERDE if status == "SINAL_FORTE" else VERMELHO

        line = (
            f"{idx:02d}. {sym:<6} "
            f"{c_score}{score:<5.1f}{RESET} "
            f"{side:<6} "
            f"{c_ema}{ema_fast:.1f}/{ema_slow:.1f}{RESET}  "
            f"{c_rsi}{rsi:.1f}{RESET}  "
            f"{c_adx}+{adx:.2f}{RESET}  "
            f"{c_mom}{mom:.2f}{RESET}  "
            f"{c_vol}{vol/1000:.0f}k{RESET}  "
            f"{price:.2f}   "
            f"{c_status}{status}{RESET}"
        )

        lines.append(line)

    footer = (
        "\n────────────────────────────────────────────────────────────────────────────────────────────────\n"
        "Mostrando sempre as 10 melhores (mesmo sem sinal) | Score 0 = não passou nos filtros\n"
        "================================================================================================\n"
    )

    return header + "\n".join(lines) + footer

def push_metrics(data: dict):
    """
    Salva métricas de execução em arquivo JSON.
    Compatível com config.METRICS_FILE.
    Seguro contra concorrência e erros silenciosos.
    """
    try:
        path = getattr(config, "METRICS_FILE", "metrics/live_metrics.json")

        # Garante diretório
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Timestamp em UTC correto (timezone-aware)
        data["timestamp"] = datetime.now(timezone.utc).isoformat()

        # Se já existe → append como lista
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    old = json.load(f)
                if not isinstance(old, list):
                    old = [old]
            except:
                old = []
        else:
            old = []

        old.append(data)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(old, f, indent=2, ensure_ascii=False)

    except Exception:
        logger.exception("Erro ao salvar métricas em push_metrics()")
