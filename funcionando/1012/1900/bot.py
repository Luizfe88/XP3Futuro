import time
import os
import signal
import sys
from datetime import datetime, time as dt_time, timezone
from typing import List, Dict

import config
import utils
from utils import logger, VERDE, VERMELHO, AMARELO, AZUL, ROXO, RESET

# ---------- CONFIGURAÇÃO ----------
CHECK_INTERVAL = getattr(config, "CHECK_INTERVAL_SLOW", 8)
TIMEFRAME = config.TIMEFRAME_MT5
SYMBOLS = list(config.SYMBOL_MAP.keys())
DEFAULT_PARAMS = config.DEFAULT_PARAMS.copy()

# Graceful shutdown
SHUTDOWN = False
def _on_sigint(signum, frame):
    global SHUTDOWN
    SHUTDOWN = True
signal.signal(signal.SIGINT, _on_sigint)
signal.signal(signal.SIGTERM, _on_sigint)


# ---------- HELPERS DE DISPLAY ----------
def fmt_money(v):
    try:
        return f"R$ {float(v):,.2f}"
    except:
        return str(v)

def print_top10(results: List[Dict]):
    txt = utils.generate_scanner_top10_elite(results)
    print(txt)


# ---------- PARAMS ADAPTIVE / LOAD ----------
def load_params_for_regime(regime: str = "DEFAULT"):
    """Carrega parâmetros adaptativos com fallback para DEFAULT_PARAMS."""
    p = DEFAULT_PARAMS.copy()
    try:
        # tenta carregar arquivo JSON por regime (se configurado)
        regime_files = {
            "STRONG_BULL": getattr(config, "PARAMS_STRONG_BULL", None),
            "BULL": getattr(config, "PARAMS_BULL", None),
            "SIDEWAYS": getattr(config, "PARAMS_SIDEWAYS", None),
            "BEAR": getattr(config, "PARAMS_BEAR", None),
            "CRISIS": getattr(config, "PARAMS_CRISIS", None),
        }
        path = regime_files.get(regime.upper())
        if path and os.path.exists(path):
            import json
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            p.update(data)
            p["regime"] = regime
            logger.info(f"Parâmetros adaptativos carregados de {path} para regime {regime}")
    except Exception:
        logger.exception("Falha ao carregar parâmetros por regime — usando DEFAULT_PARAMS")
    return p


# ---------- SCAN CYCLE ----------
def run_scan_cycle(symbols: List[str], params: dict):
    """Executa o scan paralelo e retorna resultados já prontos para exibição."""
    try:
        logger.info(f"{AZUL}Iniciando scan paralelo: {len(symbols)} símbolos{RESET}")
        results = utils.execute_parallel_scan(symbols, params, cb_active=False)
        # results é lista de dicts com 'symbol','side','score','details'
        return results
    except Exception:
        logger.exception("Erro em run_scan_cycle")
        return []


# ---------- INTEGRAÇÃO SIMPLES COM RISCO/CB ----------
def check_and_handle_circuit_breaker():
    """Exemplo de checagem de drawdown diário simples e ação."""
    try:
        # stub: aqui você colocaria sua lógica real (ex.: comparar equity vs initial daily equity)
        # retornamos False para não interromper
        return False
    except Exception:
        logger.exception("Erro ao checar circuit breaker")
        return False


# ---------- MAIN LOOP ----------
def main_loop():
    global SHUTDOWN
    logger.info(f"{VERDE}Iniciando BOT FIXED - ambiente: {config.MODE}{RESET}")

    # Se necessário, inicializa MT5 (se seu fluxo exige)
    try:
        if not mt5_initialized():
            import MetaTrader5 as mt5
            if not mt5.initialize():
                logger.warning("Falha ao inicializar MT5 — verifique a plataforma.")
            else:
                logger.info("MT5 inicializado com sucesso.")
    except Exception:
        logger.exception("Erro ao inicializar MT5")

    regime = "SIDEWAYS"  # default; você pode detectar automaticamente chamando utils.get_market_regime()
    params = load_params_for_regime(regime)

    # Fornece um primeiro scan rápido (podemos limitar símbolos)
    scan_symbols = SYMBOLS.copy()
    if len(scan_symbols) > 40:
        scan_symbols = scan_symbols[:40]  # primeiro teste mais rápido

    while not SHUTDOWN:
        try:
            # opcional: respeitar janela de operação
            now = datetime.now().time()
            if not getattr(config, "ALLOW_OUTSIDE_MARKET_HOURS", False):
            # Modo seguro (só opera no horário definido)
                if hasattr(config, "START_TIME") and hasattr(config, "END_TIME"):
                    if not (config.START_TIME <= now <= config.END_TIME):
                        logger.debug("⏳ Fora do horário de mercado — aguardando horário válido.")
                        time.sleep(min(CHECK_INTERVAL * 2, 60))
                        continue
            else:
                logger.debug("⚠️ Operação fora do horário permitida (ALLOW_OUTSIDE_MARKET_HOURS = True).")

            if check_and_handle_circuit_breaker():
                logger.critical("Circuit breaker acionado — encerrando operações")
                break

            results = run_scan_cycle(scan_symbols, params)

            # Normaliza/garante chaves para o display
            normalized = []
            for r in results:
                d = r.get("details", {}) or {}
                # Fill defaults to avoid formatting errors
                for k in ("EMA_FAST","EMA_SLOW","RSI","ADX","MOMENTUM_%","VOL_MED_20","VOLUME_ATUAL","PRECO","STATUS","SCORE"):
                    d.setdefault(k, 0 if k!="STATUS" else "SEM_DADOS")
                r["details"] = d
                normalized.append(r)

            # Exibe TOP10
            print_top10(normalized)

            # grava métricas locais (opcional)
            try:
                metrics = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "num_symbols_scanned": len(scan_symbols),
                    "top_scores": [r.get("score",0) for r in normalized[:5]],
                }
                utils.push_metrics(metrics)
            except Exception:
                logger.exception("Erro ao gravar metrics")

            # Espera e repete
            time.sleep(CHECK_INTERVAL)

        except Exception:
            logger.exception("Erro no loop principal")
            time.sleep(5)

    logger.info("Encerrando bot_fixed. Limpando conexões...")
    try:
        import MetaTrader5 as mt5
        mt5.shutdown()
    except Exception:
        pass


# ---------- UTIL: verificar se MT5 já está inicializado ----------
def mt5_initialized():
    try:
        import MetaTrader5 as mt5
        return mt5.terminal_info() is not None
    except Exception:
        return False


if __name__ == "__main__":
    main_loop()
