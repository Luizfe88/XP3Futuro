# bot.py – EXECUTOR INSTITUCIONAL B3 (V3) - SOFT STOP UNIFICADO (CORRIGIDO)

import MetaTrader5 as mt5
import time
import os
import json
import threading
import random
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np
import pandas_ta as ta
import math
import config
import utils
from utils import (logger, VERDE, VERMELHO, AMARELO, AZUL, ROXO, RESET,
                   guardiao_nuclear_posicoes_naked, pode_abrir_nova_posicao,
                   aplicar_trailing_stop_adaptativo, generate_scanner_top10_elite, execute_parallel_scan)

# ==================== CONFIGURAÇÕES GLOBAIS ====================
FORCAR_MODO_TESTE_24H = True

CURRENT_PARAMS = config.DEFAULT_PARAMS
SLIPPAGE_HISTORY = []

CB_ACTIVE = False
LAST_CB_CHECK_DAY = datetime.datetime.now().day
DAILY_START_EQUITY = 0.0

# ==================== FUNÇÕES AUXILIARES ====================

def execute_manual_test_trade(symbol="PETR4", side="COMPRA", lot=100.0, sl_mult=2.0, tp_mult=4.0):
    """
    Executa uma ordem manual de teste com SL/TP calculado por ATR.
    """
    logger.info(f"{AZUL}--- INICIANDO TESTE MANUAL: {symbol} {side} ---{RESET}")
    
    # === PREÇO ATUAL ===
    tick = mt5.symbol_info_tick(symbol)
    symbol_info = mt5.symbol_info(symbol)
    if not tick or tick.last == 0 or symbol_info is None:
        logger.error(f"{VERMELHO}Teste falhou → {symbol}: sem tick ou preço inválido.{RESET}")
        return None
    precision_digits = symbol_info.digits    
    price_current = tick.bid if side == "COMPRA" else tick.ask

    # === CÁLCULO ATR RÁPIDO ===
    try:
        rates_atr = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 20)
        if rates_atr is not None and len(rates_atr) > 14:
            df_atr = pd.DataFrame(rates_atr)
            df_atr.ta.atr(length=14, append=True)
            current_atr = df_atr['ATR_14'].iloc[-1]
        else:
            current_atr = price_current * 0.01 # Fallback
    except Exception:
        current_atr = price_current * 0.01 # Fallback
        
    logger.info(f"ATR Calculado: {current_atr:.4f}")

    # === CÁLCULO DE SL E TP ===
    if side == "COMPRA":
        sl_price = price_current - (current_atr * sl_mult)
        tp_price = price_current + (current_atr * tp_mult)
        order_type = mt5.ORDER_TYPE_BUY
    else:
        sl_price = price_current + (current_atr * sl_mult)
        tp_price = price_current - (current_atr * tp_mult)
        order_type = mt5.ORDER_TYPE_SELL
        
    sl_price = max(sl_price, 0.01)
    tp_price = max(tp_price, 0.01)

    # === ARREDONDAMENTO PARA PRECISÃO DO SÍMBOLO === <--- NOVO BLOCO
    sl_price = round(sl_price, precision_digits)
    tp_price = round(tp_price, precision_digits)
    
    # === EXECUÇÃO DA ORDEM ===
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": order_type,
        "price": price_current,
        "sl": float(sl_price),
        "tp": float(tp_price),
        "deviation": 30,
        "type_filling": mt5.ORDER_FILLING_IOC,
        "comment": "TESTE MANUAL"
    }

    result = mt5.order_send(request)
    
    if result.retcode == mt5.TRADE_RETCODE_DONE:
        logger.info(f"{VERDE}SUCESSO TESTE → {symbol} {side} {lot} lotes | Ticket: {result.order} | SL: {sl_price:.2f} | TP: {tp_price:.2f}{RESET}")
    else:
        logger.error(f"{VERMELHO}FALHA TESTE → {symbol}: {result.comment} (Erro: {result.retcode}){RESET}")
    
    return result


def load_adaptive_params():
    """Carrega parâmetros baseados no Regime de Mercado e aplica Walk-Forward."""
    global CURRENT_PARAMS
    try:
        regime_str, px_ibov, ma_ibov, vix_br = utils.get_market_regime()
    except Exception:
        # Se utils.get_market_regime falhar, usa defaults
        logger.exception("Falha ao obter regime de mercado. Usando DEFAULT.")
        regime_str, px_ibov, ma_ibov, vix_br = "DEFAULT", None, None, None

    file_map = {
        "STRONG_BULL": config.PARAMS_STRONG_BULL,
        "BULL": config.PARAMS_BULL,
        "SIDEWAYS": config.PARAMS_SIDEWAYS,
        "BEAR": config.PARAMS_BEAR,
        "CRISIS": config.PARAMS_CRISIS,
    }

    filename = file_map.get(regime_str, config.PARAMS_SIDEWAYS)

    try:
        with open(filename, 'r', encoding='utf-8') as f:
            params = json.load(f)
            CURRENT_PARAMS = params
            logger.info(f"{AZUL}Parâmetros Adaptativos ({regime_str}) carregados de {filename}{RESET}")
    except FileNotFoundError:
        logger.warning(f"Arquivo de parâmetros {filename} não encontrado. Usando DEFAULT.")
        CURRENT_PARAMS = config.DEFAULT_PARAMS
        regime_str = "DEFAULT"
    except Exception:
        logger.exception("Erro ao carregar parâmetros adaptativos. Usando DEFAULT.")
        CURRENT_PARAMS = config.DEFAULT_PARAMS
        regime_str = "DEFAULT"

    return regime_str, px_ibov, ma_ibov

def display_optimized_params():
    """Exibe os parâmetros otimizados ativos no momento."""
    global CURRENT_PARAMS
    output = f"\n=== PARÂMETROS ADAPTATIVOS ({CURRENT_PARAMS.get('regime', 'N/A')}) ==="
    output += f"\nSIDE (OPERAÇÃO)              : {CURRENT_PARAMS.get('side', 'N/A')}"
    output += f"\nEMA RÁPIDA / LENTA           : {CURRENT_PARAMS.get('ema_fast', 'N/A')}/{CURRENT_PARAMS.get('ema_slow', 'N/A')}"
    output += f"\nRSI NÍVEL                    : >{CURRENT_PARAMS.get('rsi_level', 'N/A')}"
    output += f"\nMOMENTUM MÍNIMO              : >{CURRENT_PARAMS.get('momentum_min_pct', 'N/A')}%"
    output += f"\nADX MÍNIMO                   : >{CURRENT_PARAMS.get('adx_min', 'N/A')}"
    output += f"\nSHARPE OTIMIZADO (PROXY)     : {CURRENT_PARAMS.get('sharpe_medio', 'N/A')}"
    output += f"\nSTOP LOSS (ATR Mult)         : {CURRENT_PARAMS.get('sl_atr_mult', 'N/A')}"
    output += f"\nTAKE PROFIT (ATR Mult)       : {CURRENT_PARAMS.get('tp_atr_mult', 'N/A')}"
    output += "\n--------------------------------------------------------------------------------------------------------------"
    return output

def analisar_carteira_detalhada():
    """Gera o relatório detalhado da carteira lendo as posições abertas no MT5."""
    posicoes = mt5.positions_get()

    report = "\n=== RELATÓRIO DA CARTEIRA ===\n"
    report += f"{'SÍMBOLO':<10}{'LOTE':<10}{'PREÇO MÉDIO':<15}{'PREÇO ATUAL':<15}{'PNL (R$)':<15}{'PNL (%)':<10}{'SIDE':<10}\n"
    report += "---------------------------------------------------------------------------------\n"

    pnl_total = 0.0

    if posicoes is None or (hasattr(posicoes, "__len__") and len(posicoes) == 0):
        report += "---------------------------------------------------------------------------------\n"
        report += f"TOTAL PNL FLUTUANTE{'':<47} 0.00 R$\n"
        report += "---------------------------------------------------------------------------------\n"
        return report

    for pos in posicoes:
        try:
            symbol = pos.symbol
            side = "COMPRA" if pos.type == mt5.ORDER_TYPE_BUY or pos.type == mt5.POSITION_TYPE_BUY else "VENDA"
            lote = pos.volume
            p_open = pos.price_open
            p_current = getattr(pos, "price_current", 0.0)
            profit = getattr(pos, "profit", 0.0)

            pnl_total += profit

            if p_open and p_open != 0:
                pnl_pct = (profit / (pos.volume * p_open)) * 100
            else:
                pnl_pct = 0.0

            pnl_color = VERDE if profit >= 0 else VERMELHO

            report += f"{symbol:<10}"
            report += f"{lote:<10.0f}"
            report += f"{p_open:<15.4f}"
            report += f"{p_current:<15.4f}"
            report += f"{pnl_color}{profit:<15.2f}{RESET}"
            report += f"{pnl_color}{pnl_pct:<10.2f}{RESET}"
            report += f"{side:<10}\n"
        except Exception:
            logger.exception(f"Erro ao processar posição {getattr(pos, 'symbol', 'N/A')}")

    pnl_total_color = VERDE if pnl_total >= 0 else VERMELHO
    report += "---------------------------------------------------------------------------------\n"
    report += f"TOTAL PNL FLUTUANTE{'':<47}{pnl_total_color}{pnl_total:,.2f} R${RESET}\n"
    report += "---------------------------------------------------------------------------------\n"

    return report

def execute_iceberg_order(symbol, side, lote_calc, price_current, sl_price_simulado, tp_price_simulado, max_retries=6):
    """
    Executa ordem tentando colocar SL/TP de forma atômica. Se falhar, abre sem SL/TP e tenta corrigir (modo guardião).
    Retorna o objeto result do mt5.order_send (ou None).
    """
    side_name = "COMPRA" if side == "COMPRA" else "VENDA"
    order_type = mt5.ORDER_TYPE_BUY if side == "COMPRA" else mt5.ORDER_TYPE_SELL

    # PASSO 1: tentativa atômica com SL/TP
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": float(lote_calc),
        "type": order_type,
        "price": price_current,
        "sl": round(float(sl_price_simulado), 2),
        "tp": round(float(tp_price_simulado), 2),
        "deviation": 50,
        "magic": 202612,
        "comment": "ELITE_PROTECTED",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    try:
        result = mt5.order_send(request)
    except Exception:
        logger.exception(f"Falha ao enviar ordem atômica para {symbol}")
        result = None

    if result is not None and hasattr(result, "retcode") and result.retcode == mt5.TRADE_RETCODE_DONE:
        logger.info(f"{VERDE}ABERTURA ATÔMICA COM SL/TP → {symbol} {side_name} {lote_calc:,} lotes{RESET}")
        return result

    # PASSO 2: abre sem SL/TP e corrige depois
    comment = getattr(result, "comment", "no comment") if result is not None else "no result"
    logger.warning(f"{AMARELO}Atômica falhou ({comment}). Abrindo sem SL e forçando depois...{RESET}")

    request_no_st = request.copy()
    request_no_st["sl"] = 0.0
    request_no_st["tp"] = 0.0
    request_no_st["comment"] = "ELITE_NAKED_TEMP"

    try:
        result2 = mt5.order_send(request_no_st)
    except Exception:
        logger.exception(f"Falha ao abrir posição sem SL/TP para {symbol}")
        return None

    if result2 is None or not hasattr(result2, "retcode") or result2.retcode != mt5.TRADE_RETCODE_DONE:
        logger.critical(f"{VERMELHO}FALHA TOTAL NA ABERTURA DE {symbol}: {getattr(result2, 'comment', '')}{RESET}")
        return None

    ticket = getattr(result2, "order", None)
    logger.warning(f"{AMARELO}Posição aberta SEM proteção (Ticket {ticket}). Iniciando modo GUARDIÃO...{RESET}")

    # PASSO 3: tentar colocar SL/TP via TRADE_ACTION_SLTP
    for tentativa in range(1, max_retries + 1):
        time.sleep(0.6 + tentativa * 0.4)
        modify_req = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "symbol": symbol,
            "sl": round(float(sl_price_simulado), 2),
            "tp": round(float(tp_price_simulado), 2),
        }
        try:
            res = mt5.order_send(modify_req)
        except Exception:
            logger.exception(f"Tentativa de modificar SL/TP falhou (exceção) para ticket {ticket}")
            res = None

        if res is not None and hasattr(res, "retcode") and res.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"{VERDE}GUARDIÃO ATIVADO → SL/TP colocados na tentativa {tentativa}! Ticket {ticket}{RESET}")
            return res

        logger.warning(f"Tentativa {tentativa}/{max_retries} falhou: {getattr(res, 'comment', 'no comment')}")

    logger.critical(f"{VERMELHO}GUARDIÃO FALHOU APÓS {max_retries} TENTATIVAS → {symbol} Ticket {ticket} PERMANECE DESPROTEGIDA!{RESET}")
    return result2

def scanner_paralelo(symbols_list, current_params, timeframe):
    """Executa a análise de indicadores em paralelo e retorna os resultados."""
    results = []
    MAX_THREADS = 8

    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        future_to_symbol = {
            executor.submit(utils.analyze_symbol_for_trade, symbol, timeframe, current_params): symbol
            for symbol in symbols_list
        }

        for future in as_completed(future_to_symbol):
            symbol = future_to_symbol.get(future, "N/A")
            try:
                data = future.result()
                results.append(data)
            except Exception:
                logger.exception(f'Erro ao analisar símbolo {symbol}')

    return results

def avaliar_ativo(symbol, params, side):
    """
    Avalia um ativo e retorna (sinal_ok, detalhes).
    detalhes deve conter chaves úteis para relatório.
    """
    try:
        resultado_data = utils.prepare_data_for_scan(symbol, params, lookback_days=300)
    except Exception:
        logger.exception(f"Erro ao preparar dados para {symbol}")
        return False, {"STATUS": f"{VERMELHO}Dados Ausentes{RESET}", "ATR": 0.0}

    if resultado_data is None:
        return False, {"STATUS": f"{VERMELHO}Dados Ausentes{RESET}", "ATR": 0.0}

    df, current_atr = resultado_data

    try:
        sinal_ok, detalhes = utils.check_trade_signal(df, params, side)
    except Exception:
        logger.exception(f"Erro ao checar sinal para {symbol}")
        sinal_ok = False
        detalhes = {"STATUS": "ERRO", "MOTIVO": "Erro interno"}

    detalhes = detalhes or {}
    detalhes['SIDE'] = side
    detalhes['SINAL_COMPLETO'] = bool(sinal_ok)
    detalhes['MOTIVO'] = f"APROVADO {side}" if sinal_ok else f"Falha no filtro ({side})"
    detalhes.setdefault('ATR', current_atr if current_atr is not None else 0.0)

    return bool(sinal_ok), detalhes

def load_bear_params():
    """Tenta carregar os parâmetros BEAR otimizados ou usa o default."""
    try:
        with open(config.PARAMS_BEAR, "r", encoding="utf-8") as f:
            bear_params = json.load(f)
            if 'side' not in bear_params:
                bear_params['side'] = "VENDA"
            return bear_params
    except Exception:
        logger.exception("Falha ao carregar params_bear.json. Usando DEFAULT_PARAMS_BEAR.")
        return config.DEFAULT_PARAMS_BEAR if hasattr(config, "DEFAULT_PARAMS_BEAR") else config.DEFAULT_PARAMS

def gerar_relatorio_horario():
    """Função placeholder que deveria gerar e salvar um relatório a cada hora."""
    try:
        current_hour = datetime.datetime.now().hour
        logger.info(f"{AZUL}Rotina de Relatório Horário finalizada.{RESET}")
    except Exception:
        logger.exception("Erro em gerar_relatorio_horario()")

# ==================== CICLO PRINCIPAL ====================

def ciclo_principal():
    """
    Executa o scan paralelo, gerenciamento de risco e execução com SL/TP garantido.
    """
    global CB_ACTIVE, CURRENT_PARAMS

    try:
        resultados_execucao = []
        ativos_para_scan = list(config.SYMBOL_MAP.keys())
        resultados = []
        resultados_execucao = []

        # 1. GUARDIÃO NUCLEAR – nunca posição naked
        try:
            guardiao_nuclear_posicoes_naked()
        except Exception:
            logger.exception("Erro ao executar guardiao_nuclear_posicoes_naked()")

        # 2. RELATÓRIOS
        positions = mt5.positions_get()
        try:
            if hasattr(utils, 'generate_positions_report'):
                print(utils.generate_positions_report(positions))
        except Exception:
            logger.exception("Erro ao gerar relatório de posições")

        try:
            if hasattr(utils, 'display_summary'):
                print(utils.display_summary())
        except Exception:
            logger.exception("Erro ao exibir resumo")

        # 3. TRAILING STOP ADAPTATIVO
        if positions is not None and hasattr(positions, "__len__") and len(positions) > 0:
            try:
                aplicar_trailing_stop_adaptativo(positions)
            except Exception:
                logger.exception("Erro ao aplicar trailing stop adaptativo")

        # 4. SCANNER PARALELO
        ativos = utils.get_ativos_liquidos(config.MIN_ADV_20D_BRL)

        # 5. TOP 10 ELITE (sem erro!)
        try:
            print(generate_scanner_top10_elite(resultados, top_n=10))
        except Exception:
            logger.exception("Erro ao gerar top10 elite")

        # 6. EXECUÇÃO COM CONTROLE SETORIAL
        for symbol, side, detalhes in resultados_execucao:
            try:
                pode, motivo = pode_abrir_nova_posicao(symbol)
                if not pode:
                    logger.warning(f"{AMARELO}BLOQUEADO → {symbol} | {motivo}{RESET}")
                    continue
            except Exception:
                logger.exception(f"Erro ao verificar se pode abrir nova posição para {symbol}")

        positions = mt5.positions_get()

        # === CORREÇÃO DE SEGURANÇA: APLICA SL/TP EM POSIÇÕES "NAKED" (SEM SL) ===
        if positions is not None and hasattr(positions, "__len__") and len(positions) > 0:
            for pos in positions:
                try:
                    if getattr(pos, "sl", 0.0) < 0.01:
                        logger.warning(f"{AMARELO}DETECTADO {pos.symbol} SEM SL (Ticket: {pos.ticket}). Tentando corrigir...{RESET}")

                        sl_mult = CURRENT_PARAMS.get("sl_atr_mult", 2.0)
                        tp_mult = CURRENT_PARAMS.get("tp_atr_mult", 4.0)

                        try:
                            rates = mt5.copy_rates_from_pos(pos.symbol, mt5.TIMEFRAME_M5, 0, 20)
                            if rates is not None and len(rates) > 14:
                                df_fix = pd.DataFrame(rates)
                                high = df_fix['high']
                                low = df_fix['low']
                                close = df_fix['close']
                                tr = np.maximum(high - low, np.abs(high - close.shift(1)))
                                atr_atual = tr.tail(14).mean()
                            else:
                                atr_atual = pos.price_current * 0.01
                        except Exception:
                            atr_atual = pos.price_current * 0.01

                        if pos.type == mt5.POSITION_TYPE_BUY:
                            new_sl = pos.price_current - (atr_atual * sl_mult)
                            new_tp = pos.price_current + (atr_atual * tp_mult)
                        else:
                            new_sl = pos.price_current + (atr_atual * sl_mult)
                            new_tp = pos.price_current - (atr_atual * tp_mult)

                        request = {
                            "action": mt5.TRADE_ACTION_SLTP,
                            "symbol": pos.symbol,
                            "position": pos.ticket,
                            "sl": float(new_sl),
                            "tp": float(new_tp)
                        }

                        res = mt5.order_send(request)
                        if res is not None and hasattr(res, "retcode") and res.retcode == mt5.TRADE_RETCODE_DONE:
                            logger.info(f"{VERDE}SUCESSO: SL/TP Adicionado em {pos.symbol}! SL: {new_sl:.2f}, TP: {new_tp:.2f}{RESET}")
                        else:
                            logger.error(f"{VERMELHO}FALHA ao corrigir {pos.symbol}: {getattr(res, 'comment', 'no comment')}{RESET}")

                        time.sleep(0.5)
                except Exception:
                    logger.exception(f"Erro ao processar posição {getattr(pos, 'symbol', 'N/A')}")

        MAX_LOSS_PERCENT = 6.0

        positions = mt5.positions_get()
        if positions is not None and hasattr(positions, "__len__") and len(positions) > 0:
            for pos in positions:
                try:
                    pnl_percent = 0.0
                    if getattr(pos, "price_open", 0) and getattr(pos, "volume", 0):
                        pnl_percent = (getattr(pos, "profit", 0.0) / (pos.price_open * pos.volume)) * 100

                    if pnl_percent < -MAX_LOSS_PERCENT:
                        tick = mt5.symbol_info_tick(pos.symbol)
                        if tick is None:
                            continue

                        if pos.type == mt5.POSITION_TYPE_BUY:
                            action_type = mt5.ORDER_TYPE_SELL
                            price_close = getattr(tick, "bid", None)
                            action_str = "VENDA (Zeragem)"
                        elif pos.type == mt5.POSITION_TYPE_SELL:
                            action_type = mt5.ORDER_TYPE_BUY
                            price_close = getattr(tick, "ask", None)
                            action_str = "COMPRA (Zeragem)"
                        else:
                            continue

                        if price_close is None:
                            continue

                        logger.critical(f"{VERMELHO}STOP MÁXIMO ATINGIDO: {pos.symbol} com {pnl_percent:.2f}% de perda. Zerando posição ({action_str})...{RESET}")

                        request_deal = {
                            "action": mt5.TRADE_ACTION_DEAL,
                            "symbol": pos.symbol,
                            "volume": pos.volume,
                            "type": action_type,
                            "position": pos.ticket,
                            "price": price_close,
                            "deviation": 30,
                            "type_filling": mt5.ORDER_FILLING_IOC,
                        }

                        res = mt5.order_send(request_deal)
                        if res is not None and hasattr(res, "retcode") and res.retcode == mt5.TRADE_RETCODE_DONE:
                            logger.critical(f"{VERMELHO}SUCESSO: {pos.symbol} zerada por Stop Máximo. PnL: R$ {pos.profit:,.2f}{RESET}")
                        else:
                            logger.error(f"{VERMELHO}FALHA ao zerar {pos.symbol} por Stop Máximo: {getattr(res, 'comment', 'no comment')}{RESET}")

                        time.sleep(1)
                except Exception:
                    logger.exception(f"Erro ao avaliar stop máximo para {getattr(pos, 'symbol', 'N/A')}")

        # 1. CARREGAR PARÂMETROS ADAPTATIVOS
        regime_str, px_ibov, ma_ibov = load_adaptive_params()
        bull_params = CURRENT_PARAMS
        bear_params = load_bear_params()

        # 2. RELATÓRIOS E DISPLAY
        positions = mt5.positions_get()
        try:
            print(utils.generate_positions_report(positions) if hasattr(utils, 'generate_positions_report') else "")
        except Exception:
            logger.exception("Erro ao imprimir generate_positions_report")

        if positions is not None and hasattr(positions, "__len__") and len(positions) > 0:
            try:
                aplicar_trailing_stop_adaptativo(positions)
            except Exception:
                logger.exception("Erro ao aplicar trailing stop adaptativo (segunda chamada)")

        try:
            print(utils.display_summary() if hasattr(utils, 'display_summary') else "")
        except Exception:
            logger.exception("Erro ao imprimir display_summary")

        try:
            print(display_optimized_params())
        except Exception:
            logger.exception("Erro ao imprimir parâmetros otimizados")

        try:
            print(analisar_carteira_detalhada())
        except Exception:
            logger.exception("Erro ao imprimir analisar_carteira_detalhada")

        # 3. CIRCUIT BREAKER (SOFT STOP)
        if CB_ACTIVE:
            logger.critical(f"{VERMELHO}CIRCUIT BREAKER ATIVO — NENHUMA NOVA ORDEM SERÁ EXECUTADA{RESET}")
            return

        # 4. PREPARAÇÃO E EXECUÇÃO DO SCANNER (para o relatório)
        ativos_para_scan = utils.get_ativos_liquidos(config.MIN_ADV_20D_BRL)
        try:
            simbolos_analisados = utils.execute_parallel_scan(
                ativos_para_scan,
                CURRENT_PARAMS,
                CB_ACTIVE
            )
        except Exception:
            logger.exception("Erro em execute_parallel_scan()")
            simbolos_analisados = []

        # 5. Executa o scanner em paralelo para COMPRA (usa bull_params como base)
        logger.info(f"{AZUL}Iniciando varredura em {len(ativos_para_scan)} ativos...{RESET}")
        try:
            scanner_results = generate_scanner_top10_elite(simbolos_analisados)
        except Exception:
            logger.exception("Erro ao gerar scanner_results")
            scanner_results = ""

        # 6. Geração e Impressão do Relatório do Scanner (Top 20 e Motivos)
        try:
            print(scanner_results)
        except Exception:
            logger.exception("Erro ao imprimir scanner_results")

        # 7. FILTROS FINAIS E GESTÃO DE PORTFOLIO
        resultados_execucao = []

        posicoes_abertas = mt5.positions_get()
        num_posicoes = len(posicoes_abertas) if (posicoes_abertas is not None and hasattr(posicoes_abertas, "__len__")) else 0

        if num_posicoes >= config.MAX_POSITIONS:
            logger.warning(f"{ROXO}LIMITE DE POSIÇÕES ATINGIDO ({num_posicoes}/{config.MAX_POSITIONS}). ABORTANDO NOVAS EXECUÇÕES.{RESET}")
            resultados_execucao = []

        elif len(resultados_execucao) > config.MAX_TRADES_PER_CYCLE:
            logger.warning(f"{ROXO}LIMITE DE RATE LIMIT ATINGIDO ({len(resultados_execucao)}/{config.MAX_TRADES_PER_CYCLE}). Reduzindo lista...{RESET}")
            resultados_execucao = resultados_execucao[:config.MAX_TRADES_PER_CYCLE]

        # 8. EXECUÇÃO DAS ORDENS
        for symbol, side, detalhes in resultados_execucao:
            try:
                if not utils.is_market_open(symbol):
                    logger.warning(f"{AMARELO}ABORTAR {symbol}: Mercado Fechado ou em Leilão.{RESET}")
                    continue

                tick = mt5.symbol_info_tick(symbol)
                if tick is None or getattr(tick, "last", 0) == 0:
                    logger.warning(f"Tentativa falhou → {symbol}: sem tick")
                    continue

                price_current = getattr(tick, "bid", None) if side == "COMPRA" else getattr(tick, "ask", None)
                if price_current is None or price_current == 0:
                    logger.warning(f"{AMARELO}Preço inválido para {symbol}. Pulando.{RESET}")
                    continue

                # ATR
                try:
                    rates_atr = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 20)
                    if rates_atr is not None and len(rates_atr) > 14:
                        df_atr = pd.DataFrame(rates_atr)
                        df_atr.ta.atr(length=14, append=True)
                        current_atr = df_atr['ATR_14'].iloc[-1]
                    else:
                        current_atr = price_current * 0.01
                except Exception:
                    logger.exception(f"Erro ao calcular ATR para {symbol}")
                    current_atr = price_current * 0.01

                sl_mult = CURRENT_PARAMS.get("sl_atr_mult", 2.0)
                tp_mult = CURRENT_PARAMS.get("tp_atr_mult", 4.0)

                if side == "COMPRA":
                    sl_price = price_current - (current_atr * sl_mult)
                    tp_price = price_current + (current_atr * tp_mult)
                else:
                    sl_price = price_current + (current_atr * sl_mult)
                    tp_price = price_current - (current_atr * tp_mult)

                sl_price = max(sl_price, 0.01)
                tp_price = max(tp_price, 0.01)

                acc = mt5.account_info()
                if acc is None or getattr(acc, "equity", 0) <= 0:
                    logger.error("Não foi possível obter equity da conta")
                    continue

                risco_reais = acc.equity * 0.01
                distancia_sl = abs(price_current - sl_price)

                if distancia_sl < 0.01:
                    logger.warning(f"{symbol} → Distância SL muito curta. Pulando.")
                    continue

                lote_bruto_risco = risco_reais / (distancia_sl * 100)
                lote_base = int(math.floor(lote_bruto_risco / 100) * 100)
                lote_base = max(100, lote_base)

                if lote_base < 100:
                    logger.warning(f"{symbol} → Lote calculado muito baixo ({lote_base}). Pulando.")
                    continue

                order_type = mt5.ORDER_TYPE_BUY if side == "COMPRA" else mt5.ORDER_TYPE_SELL
                lote = lote_base

                lote_to_check = lote
                while lote_to_check >= 100:
                    check_request = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "symbol": symbol,
                        "volume": float(lote_to_check),
                        "type": order_type,
                        "price": price_current,
                        "deviation": 30,
                    }

                    try:
                        check_result = mt5.order_check(check_request)
                    except Exception:
                        logger.exception(f"Erro em order_check para {symbol}")
                        check_result = None

                    if check_result is not None and hasattr(check_result, "retcode") and (check_result.retcode == mt5.TRADE_RETCODE_OK or check_result.retcode == 10009):
                        lote = lote_to_check
                        break
                    elif check_result is not None and hasattr(check_result, "retcode") and check_result.retcode == 10019:
                        logger.warning(f"{AMARELO}Margem insuficiente (10019) para {symbol} Lote {lote_to_check}. Tentando Lote {lote_to_check - 100}.{RESET}")
                        lote_to_check -= 100
                    else:
                        logger.error(f"{VERMELHO}Erro de checagem de ordem {symbol}: {getattr(check_result, 'retcode', 'N/A')} -> {getattr(check_result, 'comment', 'no comment')}. Pulando execução.{RESET}")
                        lote_to_check = 0

                if lote_to_check < 100:
                    logger.warning(f"{AMARELO}{symbol} → Lote mínimo de 100 não suportado pela margem. Pulando execução.{RESET}")
                    continue

                logger.info(f"{AZUL}EXECUTANDO → {symbol} {side} | Lote: {lote:,} | Preço: {price_current:.2f} | SL: {sl_price:.2f} | TP: {tp_price:.2f}{RESET}")

                result = execute_iceberg_order(
                    symbol=symbol,
                    side=side,
                    lote_calc=lote,
                    price_current=price_current,
                    sl_price_simulado=sl_price,
                    tp_price_simulado=tp_price
                )

                if result is not None and hasattr(result, "retcode") and result.retcode == mt5.TRADE_RETCODE_DONE:
                    logger.info(f"{VERDE}SUCESSO TOTAL → {symbol} {side} {lote:,} lotes | Ticket: {getattr(result, 'order', 'N/A')} | SL/TP COLOCADOS{RESET}")
                else:
                    logger.error(f"{VERMELHO}Falha ao executar {symbol} {side}{RESET}")

            except Exception:
                logger.exception(f"Erro ao executar ordem para {symbol}")

        logger.info(f"{AZUL}Ciclo concluído. Próxima varredura em {config.CHECK_INTERVAL_SLOW}s...{RESET}")

        gerar_relatorio_horario()

    except Exception:
        logger.exception("Exceção não tratada no ciclo_principal()")

# ==================== FUNÇÃO MAIN (LOOP) ====================

def main():
    global CB_ACTIVE, LAST_CB_CHECK_DAY, DAILY_START_EQUITY

    # 1. TENTA CONEXÃO SILENCIOSA COM MT5 ABERTO
    if not mt5.initialize():
        logger.warning("Falha na conexão silenciosa. Tentando inicialização completa...")
        if not mt5.initialize(login=config.LOGIN, password=config.PASSWORD, server=config.SERVER):
            logger.critical(f"Falha na inicialização do MT5 (Login/Senha). Erro: {mt5.last_error()}. Encerrando.")
            time.sleep(10)
            return

    logger.info(f"{VERDE}Conexão MT5 Estabelecida com sucesso!{RESET}")

    while True:
        try:
            current_time = datetime.datetime.now().time()
            acc_info = mt5.account_info()

            # Reset Diário
            if datetime.datetime.now().day != LAST_CB_CHECK_DAY:
                CB_ACTIVE = False
                LAST_CB_CHECK_DAY = datetime.datetime.now().day
                DAILY_START_EQUITY = 0.0
                logger.info(f"{AZUL}--- RESET DIÁRIO CONCLUÍDO ---{RESET}")

            if DAILY_START_EQUITY == 0.0 and acc_info is not None:
                DAILY_START_EQUITY = getattr(acc_info, "equity", 0.0)
                logger.info(f"{AZUL}📈 Equity Inicial do Dia Definido: R$ {DAILY_START_EQUITY:,.2f}{RESET}")

            # Checagem do Circuit Breaker (Soft Stop Unificado)
            tick_data = mt5.symbol_info_tick(config.IBOV_SYMBOL)
            # DEBUG: Força a flag para False para garantir o teste do scanner
            CB_ACTIVE = False

            # Verifica Horário de Operação → MODO TESTE 24H (COMENTE OU DESATIVE PARA PRODUÇÃO)
            FORCAR_OPERACAO_24H = False

            if not FORCAR_MODO_TESTE_24H:
                if not (config.START_TIME <= current_time <= config.END_TIME):
                    print(f"Fora do horário B3: {current_time} | Aguardando...", end='\r')
                    time.sleep(30)
                    continue
                else:
                    if not utils.is_market_open("WINZ25"):
                        print(f"{AMARELO}MERCADO FECHADO → MODO SIMULAÇÃO 24H ATIVO (sem ordens reais){RESET}", end='\r')
                        os.system('cls' if os.name == 'nt' else 'clear')
                        ciclo_principal()
                        time.sleep(10)
                        continue

            # Guardião de risco: Soft Stop Unificado
            acc_info = mt5.account_info()
            if acc_info is None:
                logger.error("Falha ao obter dados da conta. Pulando ciclo.")
                time.sleep(config.CHECK_INTERVAL_SLOW)
                continue

            positions_list = mt5.positions_get()
            # Aqui você pode calcular drawdown diário, VaR, etc. e setar CB_ACTIVE = True se necessário.
            # Exemplo simples (placeholder):
            try:
                unrealized = sum([getattr(p, "profit", 0.0) for p in positions_list]) if (positions_list is not None and hasattr(positions_list, "__len__")) else 0.0
                drawdown_pct = 0.0
                if DAILY_START_EQUITY and DAILY_START_EQUITY > 0:
                    drawdown_pct = abs(unrealized) / DAILY_START_EQUITY * 100
                logger.info(f"Monitoramento Soft Stop: Drawdown Diário {drawdown_pct:.2f}% | Limite: {config.MAX_DAILY_DRAWDOWN_PERCENT:.2f}%")
                if drawdown_pct >= config.MAX_DAILY_DRAWDOWN_PERCENT:
                    CB_ACTIVE = True
                    logger.critical(f"{VERMELHO}Circuit Breaker ativado por drawdown diário ({drawdown_pct:.2f}%).{RESET}")
            except Exception:
                logger.exception("Erro ao calcular drawdown diário")

            # Executa ciclo principal
            ciclo_principal()

            time.sleep(config.CHECK_INTERVAL_SLOW)
        
        except Exception:
            logger.exception("Exceção não tratada no Loop Principal")
            time.sleep(config.CHECK_INTERVAL_SLOW)

# TENTATIVA DE COMPRA: PETR4, 100 lotes (Lembre-se: o lote mínimo é geralmente 100)
        #execute_manual_test_trade(symbol="PETR4", side="COMPRA", lot=100.0)            

if __name__ == "__main__":
    main()
