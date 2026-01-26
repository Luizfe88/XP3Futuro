# utils.py – VERSÃO 100% COMPLETA, CORRIGIDA E TURBINADA (SAÍDA RÁPIDA XP – DEZ/2025)
import logging
import csv
import os
import threading
import time
import winsound
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict

import MetaTrader5 as mt5
import pandas as pd
import pandas_ta as ta

import config
from colorama import init, Fore, Style

init(autoreset=True)

# CORES
VERDE = Fore.GREEN + Style.BRIGHT
VERMELHO = Fore.RED + Style.BRIGHT
AMARELO = Fore.YELLOW + Style.BRIGHT
AZUL = Fore.CYAN + Style.BRIGHT
ROXO = Fore.MAGENTA + Style.BRIGHT
BRANCO = Fore.WHITE + Style.BRIGHT
RESET = Style.RESET_ALL


# LOGGER
def setup_logger():
    logger = logging.getLogger("BotElite2026")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        os.makedirs("logs", exist_ok=True)
        fh = logging.FileHandler(config.LOG_FILE, encoding="utf-8")
        ch = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s")
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger


logger = setup_logger()


# ====================== FUNÇÕES BÁSICAS ======================
def get_balance():
    acc = mt5.account_info()
    return float(acc.balance) if acc else 0.0


def calcular_volume_inteligente(symbol):
    tick = mt5.symbol_info_tick(symbol)
    if not tick or tick.ask <= 0:
        return 0
    preco = tick.ask
    saldo = get_balance()
    if saldo < 3000:
        return 0

    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_D1, 0, config.ATR_PERIOD + 10)
    if rates is None or len(rates) < config.ATR_PERIOD:
        atr = preco * 0.025
    else:
        df = pd.DataFrame(rates)
        tr = pd.concat(
            [
                df["high"] - df["low"],
                abs(df["high"] - df["close"].shift()),
                abs(df["low"] - df["close"].shift()),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(config.ATR_PERIOD).mean().iloc[-1]
        if pd.isna(atr):
            atr = preco * 0.025

    risco_reais = saldo * (config.RISCO_PORCENTO_DO_CAPITAL / 100)
    perda_por_acao = max(preco * config.STOP_LOSS_PCT, atr * config.ATR_MULTIPLIER)
    acoes_risco = risco_reais / perda_por_acao
    acoes_capital = config.CAPITAL_MAX_POR_ATIVO_REAIS / preco
    acoes_desejadas = min(acoes_risco, acoes_capital)

    volume = (int(acoes_desejadas) // 100) * 100
    if volume == 0 and acoes_desejadas >= 80:
        volume = 100
    return max(0, volume)


def get_setor(symbol):
    base = symbol.replace(config.CORRETORA_SUFFIX, "")
    return config.SETORES.get(base, "OUTROS")


def contar_setor_atual():
    contagem = defaultdict(int)
    positions = mt5.positions_get() or []
    for p in positions:
        contagem[get_setor(p.symbol)] += 1
    return contagem


def ibov_ok():
    symbol = config.IBOV_SYMBOL
    tick = mt5.symbol_info_tick(symbol)
    if not tick or tick.last <= 0:
        return True
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_D1, 0, 1)
    if not rates:
        return True
    open_price = rates[0]["open"]
    if open_price <= 0:
        return True
    return (tick.last - open_price) / open_price >= config.IBOV_MAX_DROP_PCT


def tocar_som_compra():
    try:
        winsound.Beep(1200, 600)
    except:
        pass


def is_trading_time():  # <<< ESSA FUNÇÃO ESTAVA FALTANDO!
    now = datetime.now().time()
    return config.START_TIME <= now <= config.END_TIME


# ====================== DADOS TÉCNICOS ======================
def get_asset_technical_data(symbol):
    tick = mt5.symbol_info_tick(symbol)
    if not tick or tick.ask <= 0 or tick.bid <= 0:
        return None, "Tick inválido"

    rates_d1 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_D1, 0, 70)
    rates_m5 = mt5.copy_rates_range(
        symbol,
        config.TIMEFRAME_MT5,
        datetime.now() - timedelta(days=12),
        datetime.now(),
    )

    if rates_d1 is None or len(rates_d1) < 50:
        return None, "Sem dados D1 suficientes"
    if rates_m5 is None or len(rates_m5) < 60:
        return None, "Sem dados M5 suficientes"

    df_m5 = pd.DataFrame(rates_m5)
    df_d1 = pd.DataFrame(rates_d1)

    price_last = tick.last if (hasattr(tick, "last") and tick.last > 0) else tick.ask
    prev_close_d1 = float(df_d1.iloc[-2]["close"])
    momentum = ((price_last / prev_close_d1) - 1) * 100

    df_m5["EMA_FAST"] = ta.ema(df_m5["close"], length=config.EMA_FAST)
    df_m5["EMA_SLOW"] = ta.ema(df_m5["close"], length=config.EMA_SLOW)
    df_m5["RSI"] = ta.rsi(df_m5["close"], length=config.RSI_PERIOD)
    df_m5["VOL_MA"] = ta.sma(df_m5["tick_volume"], length=config.VOLUME_MA_PERIOD)

    last_row = df_m5.iloc[-2]
    if any(
        pd.isna(
            [
                last_row["EMA_FAST"],
                last_row["EMA_SLOW"],
                last_row["RSI"],
                last_row["VOL_MA"],
            ]
        )
    ):
        return None, "Indicadores NaN"

    ema_rapida = ta.ema(df_d1["close"], config.EMA_TENDENCIA_RAPIDA).iloc[-1]
    ema_lenta = ta.ema(df_d1["close"], config.EMA_TENDENCIA_LENTA).iloc[-1]
    tendencia_forte = ema_rapida > ema_lenta and price_last > ema_rapida

    data = {
        "symbol": symbol,
        "price": float(price_last),
        "RSI": float(last_row["RSI"]),
        "EMA_FAST": float(last_row["EMA_FAST"]),
        "EMA_SLOW": float(last_row["EMA_SLOW"]),
        "momentum": float(momentum),
        "curr_vol": int(df_m5.iloc[-1]["tick_volume"]),
        "avg_vol": float(last_row["VOL_MA"]),
        "spread": float(tick.ask - tick.bid),
        "tendencia_d1": tendencia_forte,
    }
    return data, None


# ====================== REGISTRO E ZERAR ======================
def register_trade(symbol, side, price, volume, result=""):
    os.makedirs("trades", exist_ok=True)
    with open(config.CSV_FILE, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(
            [
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                symbol,
                side,
                f"{price:.3f}",
                volume,
                result,
            ]
        )


def close_all_positions():
    positions = mt5.positions_get() or []
    for pos in positions:
        tick = mt5.symbol_info_tick(pos.symbol)
        if not tick:
            continue
        order_type = (
            mt5.ORDER_TYPE_SELL
            if pos.type == mt5.ORDER_TYPE_BUY
            else mt5.ORDER_TYPE_BUY
        )
        price = tick.bid if order_type == mt5.ORDER_TYPE_SELL else tick.ask
        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": pos.symbol,
            "volume": pos.volume,
            "type": order_type,
            "position": pos.ticket,
            "price": price,
            "deviation": 300,
            "magic": 999999,
            "type_filling": mt5.ORDER_FILLING_RETURN,
            "type_time": mt5.ORDER_TIME_GTC,
        }
        r = mt5.order_send(req)
        if r and r.retcode in [mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_PLACED]:
            logger.warning(f"{ROXO}ZEROU → {pos.symbol} | {pos.volume} ações{RESET}")
            register_trade(pos.symbol, "CLOSE_ALL", price, pos.volume, "PANIC")


def get_open_positions_display():
    positions = mt5.positions_get() or []
    if not positions:
        return f"{AZUL}Nenhuma posição aberta{RESET}"
    lines = []
    for pos in positions:
        if pos.magic not in [20251230, 999999, 777777]:
            continue
        try:
            opened_at = datetime.fromtimestamp(pos.time)
            dur = str(datetime.now() - opened_at).split(".")[0]
        except:
            dur = "???"
        tick = mt5.symbol_info_tick(pos.symbol)
        if not tick or tick.last <= 0:
            status = f"{AMARELO}Sem tick{RESET}"
        else:
            preco_atual = tick.last if tick.last > 0 else tick.bid
            lucro_pct = (preco_atual - pos.price_open) / pos.price_open * 100
            explicacoes = []
            if lucro_pct < config.TAKE_PROFIT_PCT * 100:
                falta = config.TAKE_PROFIT_PCT * 100 - lucro_pct
                explicacoes.append(
                    f"TP +{config.TAKE_PROFIT_PCT*100:.1f}% (falta +{falta:.1f}%)"
                )
            else:
                explicacoes.append(f"{VERDE}TP BATIDO!{RESET}")
            if lucro_pct < config.TRAILING_ACTIVATION_PCT * 100:
                falta_tr = config.TRAILING_ACTIVATION_PCT * 100 - lucro_pct
                explicacoes.append(
                    f"Trail +{config.TRAILING_ACTIVATION_PCT*100:.1f}% (falta +{falta_tr:.1f}%)"
                )
            else:
                explicacoes.append(f"Trailing ativo")
            if pos.sl:
                dist = (preco_atual - pos.sl) / pos.sl * 100
                if dist > 0:
                    explicacoes.append(f"Stop {dist:.1f}% abaixo")
                else:
                    explicacoes.append(f"{VERMEL呵呵}STOP ATINGIDO!{RESET}")
            status = " | ".join(explicacoes)

        base = pos.volume * pos.price_open
        pnl_pct = (pos.profit / base) * 100 if base > 1 else 0.0
        cor_pnl = VERDE if pos.profit >= 0 else VERMELHO
        side = "LONG" if pos.type == mt5.ORDER_TYPE_BUY else "SHORT"
        lines.append(
            f"{BRANCO}{pos.symbol:<10}{RESET} | {side:<5} | {int(pos.volume):>4} ações | "
            f"Entrada {pos.price_open:.2f} → {cor_pnl}{pnl_pct:+.2f}% (R${pos.profit:+.0f}){RESET} | {dur} | {status}"
        )
    return "\n".join(lines)


# ====================== TRAILING STOP + SAÍDA RÁPIDA ======================
trailing_running = False


def trailing_stop_thread():
    global trailing_running
    logger.info(f"{VERDE}Trailing Stop + Saída Rápida iniciado{RESET}")
    trailing_running = True
    while trailing_running:
        try:
            positions = mt5.positions_get()
            if not positions:
                time.sleep(0.3)
                continue

            for pos in positions:
                if pos.magic not in [20251230, 999999, 777777]:
                    continue

                tick = mt5.symbol_info_tick(pos.symbol)
                if not tick or tick.last <= 0:
                    continue

                preco_atual = (
                    tick.last
                    if tick.last > 0
                    else (tick.bid if pos.type == 0 else tick.ask)
                )
                lucro_pct = (preco_atual - pos.price_open) / pos.price_open * 100

                if lucro_pct >= config.TAKE_PROFIT_PCT * 100:
                    fechar_com_motivo(pos, tick, "TP BATIDO")
                    continue

                if pos.sl and preco_atual <= pos.sl + 0.01:
                    fechar_com_motivo(pos, tick, "STOP LOSS")
                    continue

                if lucro_pct >= config.TRAILING_ACTIVATION_PCT * 100:
                    novo_sl = round(preco_atual * (1 - config.TRAILING_STOP_PCT), 2)
                    if not pos.sl or novo_sl > pos.sl + 0.01:
                        mt5.order_send(
                            {
                                "action": mt5.TRADE_ACTION_SLTP,
                                "position": pos.ticket,
                                "sl": novo_sl,
                                "tp": pos.tp,
                            }
                        )
                        logger.info(
                            f"{ROXO}TRAILING → {pos.symbol} | Novo SL R${novo_sl}{RESET}"
                        )
                    if preco_atual <= novo_sl + 0.01:
                        fechar_com_motivo(pos, tick, "TRAILING STOP")

        except Exception as e:
            logger.error(f"Erro no trailing: {e}")
        time.sleep(0.3)  # 4× mais rápido que antes


def fechar_com_motivo(pos, tick, motivo, max_tentativas=5):
    symbol = pos.symbol
    volume_restante = pos.volume
    ticket = pos.ticket
    is_long = pos.type == mt5.ORDER_TYPE_BUY

    for tentativa in range(1, max_tentativas + 1):
        if volume_restante <= 0:
            break

        tick_atual = mt5.symbol_info_tick(symbol)
        if not tick_atual:
            time.sleep(0.1)
            continue

        preco = tick_atual.bid if is_long else tick_atual.ask
        tipo_ordem = mt5.ORDER_TYPE_SELL if is_long else mt5.ORDER_TYPE_BUY

        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume_restante,
            "type": tipo_ordem,
            "position": ticket,
            "price": preco,
            "deviation": 300,
            "type_filling": mt5.ORDER_FILLING_RETURN,
            "type_time": mt5.ORDER_TIME_GTC,
            "magic": 999999,
            "comment": f"SAIDA_{motivo.upper()}",
        }

        r = mt5.order_send(req)

        if r and r.retcode in [mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_PLACED]:
            executado = getattr(r, "volume", volume_restante)
            volume_restante -= executado
            logger.warning(
                f"{VERDE}SAÍDA → {symbol} | {executado} ações | {motivo} (tent.{tentativa}){RESET}"
            )
        else:
            erro = r.comment if r else "None"
            logger.error(
                f"{VERMELHO}Falha saída {symbol} | retcode {getattr(r, 'retcode', 'N/A')} | {erro}{RESET}"
            )
            time.sleep(0.12)

    if volume_restante <= 0:
        logger.warning(f"{VERDE}POSIÇÃO 100% FECHADA → {symbol} | {motivo}{RESET}")
        register_trade(symbol, "SELL", preco, pos.volume, motivo)
    else:
        logger.error(
            f"{VERMELHO}FALHA TOTAL → {symbol} | ainda aberto {volume_restante} ações{RESET}"
        )


# INICIA O TRAILING AUTOMATICAMENTE
trailing_running = True
