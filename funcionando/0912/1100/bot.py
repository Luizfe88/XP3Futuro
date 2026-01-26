# bot.py – EXECUTOR INSTITUCIONAL B3 (V3) - SOFT STOP UNIFICADO

import MetaTrader5 as mt5
import time
import os
import json
import threading
import random
import datetime 
from concurrent.futures import ThreadPoolExecutor, as_completed # <--- CORREÇÃO!
import pandas as pd
import numpy as np
import pandas_ta as ta # Import necessário para .ta.ema e .ta.rsi
import math
import config
import utils
from utils import logger, Fore, Style, VERDE, AMARELO, VERMELHO, AZUL, ROXO, RESET, generate_positions_report, analyze_symbol_for_trade, generate_scanner_report

# ==================== VARIÁVEIS GLOBAIS DE CONTROLE ====================
CURRENT_PARAMS = config.DEFAULT_PARAMS
SLIPPAGE_HISTORY = [] 

# NOVOS: Flag Global para Circuit Breaker e rastreio diário
CB_ACTIVE = False # Se True, impede novas compras/vendas
LAST_CB_CHECK_DAY = datetime.datetime.now().day # Controla o reset diário
DAILY_START_EQUITY = 0.0 # Equity no início do dia
# =======================================================================

# ==================== FUNÇÕES AUXILIARES EXISTENTES (MANUTIDAS) ====================

def load_adaptive_params():
    """Carrega parâmetros baseados no Regime de Mercado E aplica Walk-Forward."""
    global CURRENT_PARAMS
    # Assume que esta função está implementada em utils e retorna o regime e indicadores IBOV
    regime_str, px_ibov, ma_ibov, vix_br = utils.get_market_regime()
    
    file_map = {
        "STRONG_BULL": config.PARAMS_STRONG_BULL,
        "BULL": config.PARAMS_BULL,
        "SIDEWAYS": config.PARAMS_SIDEWAYS,
        "BEAR": config.PARAMS_BEAR,
        "CRISIS": config.PARAMS_CRISIS,
    }
    
    filename = file_map.get(regime_str, config.PARAMS_SIDEWAYS)
    
    try:
        with open(filename, 'r') as f:
            params = json.load(f)
            CURRENT_PARAMS = params
            logger.info(f"{AZUL}Parâmetros Adaptativos ({regime_str}) carregados de {filename}{RESET}")
    except FileNotFoundError:
        logger.warning(f"Arquivo de parâmetros {filename} não encontrado. Usando DEFAULT.")
        CURRENT_PARAMS = config.DEFAULT_PARAMS
        regime_str = "DEFAULT"
        
    return regime_str, px_ibov, ma_ibov


def display_optimized_params():
    """Exibe os parâmetros otimizados ativos no momento."""
    global CURRENT_PARAMS
    # Formato ajustado para a saída desejada
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


# --- Trecho do bot.py (Substituir a função analisar_carteira_detalhada) ---
def analisar_carteira_detalhada():
    """Gera o relatório detalhado da carteira lendo as posições abertas no MT5."""
    
    # 1. Busca todas as posições abertas
    posicoes = mt5.positions_get()
    
    report = "\n=== RELATÓRIO DA CARTEIRA ===\n"
    report += f"{'SÍMBOLO':<10}{'LOTE':<10}{'PREÇO MÉDIO':<15}{'PREÇO ATUAL':<15}{'PNL (R$)':<15}{'PNL (%)':<10}{'SIDE':<10}\n"
    report += "---------------------------------------------------------------------------------\n"
    
    pnl_total = 0.0

    if posicoes is None or len(posicoes) == 0:
        report += "---------------------------------------------------------------------------------\n"
        report += f"TOTAL PNL FLUTUANTE{'':<47} 0.00 R$\n"
        report += "---------------------------------------------------------------------------------\n"
        # Se zerada, retorna o relatório de 0 posições
        return report

    # 2. Processa as posições e calcula PNL
    for pos in posicoes:
        symbol = pos.symbol
        side = "COMPRA" if pos.type == mt5.ORDER_TYPE_BUY else "VENDA"
        lote = pos.volume
        p_open = pos.price_open
        p_current = pos.price_current
        profit = pos.profit
        
        pnl_total += profit
        
        # Cálculo do PNL percentual (evita divisão por zero)
        if pos.price_open != 0:
            pnl_pct = (profit / (pos.volume * pos.price_open)) * 100
        else:
            pnl_pct = 0.0

        # Aplica cores ao PNL
        pnl_color = VERDE if profit >= 0 else VERMELHO
        
        # Formata a linha do relatório
        report += f"{symbol:<10}"
        report += f"{lote:<10.0f}"
        report += f"{p_open:<15.4f}"
        report += f"{p_current:<15.4f}"
        report += f"{pnl_color}{profit:<15.2f}{RESET}"
        report += f"{pnl_color}{pnl_pct:<10.2f}{RESET}"
        report += f"{side:<10}\n"

    # 3. Adiciona o total
    pnl_total_color = VERDE if pnl_total >= 0 else VERMELHO
    report += "---------------------------------------------------------------------------------\n"
    report += f"TOTAL PNL FLUTUANTE{'':<47}{pnl_total_color}{pnl_total:,.2f} R${RESET}\n"
    report += "---------------------------------------------------------------------------------\n"
    
    return report

def execute_iceberg_order(symbol, side, lote_calc, price_current, sl_price_simulado, tp_price_simulado):
    """
    Executa ordem a mercado e coloca SL/TP em seguida com verificação robusta.
    """
    if side == "COMPRA":
        order_type = mt5.ORDER_TYPE_BUY
        side_name = "COMPRA"
    else:
        order_type = mt5.ORDER_TYPE_SELL
        side_name = "VENDA"

    # === 1. ENVIA ORDEM A MERCADO ===
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": float(lote_calc), # Garante float
        "type": order_type,
        "price": price_current,
        "deviation": 30,
        "type_filling": mt5.ORDER_FILLING_IOC,
        "type_time": mt5.ORDER_TIME_GTC,
        "comment": "BOT ELITE 2026",
    }

    result = mt5.order_send(request)

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logger.error(f"{VERMELHO}FALHA AO ABRIR {symbol}: {result.retcode} -> {result.comment}{RESET}")
        return None

    logger.info(f"{VERDE}EXECUTADO -> {symbol} {side_name} {lote_calc} @ {price_current:.2f}{RESET}")

    # === 2. ROTINA DE PERSISTÊNCIA DE SL/TP ===
    # Tenta colocar o SL por até 5 tentativas, aumentando o tempo de espera
    sl = round(sl_price_simulado, 2)
    tp = round(tp_price_simulado, 2)
    
    # O ticket da ordem geralmente vira o ticket da posição, mas vamos confirmar
    position_ticket = result.order 
    
    for i in range(5):
        time.sleep(0.5 + (i * 0.5)) # Espera progressiva: 0.5s, 1.0s, 1.5s...
        
        # Verifica se a posição existe
        positions = mt5.positions_get(ticket=position_ticket)
        if positions is None or len(positions) == 0:
            # Tenta buscar pelo símbolo se não achar pelo ticket
            positions = mt5.positions_get(symbol=symbol)
            if positions:
                # Pega a última posição aberta deste símbolo
                position_ticket = positions[-1].ticket
            else:
                logger.warning(f"Tentativa {i+1}: Posição {symbol} ainda não encontrada...")
                continue

        modify_request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": symbol,
            "position": position_ticket,
            "sl": sl,
            "tp": tp,
        }

        modify_result = mt5.order_send(modify_request)

        if modify_result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"{VERDE}SL/TP DEFINIDOS COM SUCESSO -> SL: {sl} | TP: {tp}{RESET}")
            return result # Sucesso! Sai da função
        
        elif modify_result.retcode == 10016: # 10016 = Stops inválidos (muito perto do preço)
             logger.warning(f"Stops muito curtos para {symbol}. Ajustando...")
             # Aqui você poderia afastar o stop se quisesse, ou apenas tentar de novo
        
        logger.warning(f"Tentativa {i+1} falhou ({modify_result.retcode}). Retentando...")

    logger.critical(f"{VERMELHO}ERRO CRÍTICO: Posição {symbol} ficou SEM SL após 5 tentativas!{RESET}")
    return result

# --- Em bot.py: Adicione esta função ---
def scanner_paralelo(symbols_list, current_params, timeframe):
    """Executa a análise de indicadores em paralelo e retorna os resultados."""
    results = []
    MAX_THREADS = 8 
    
    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        # Envia todas as tarefas de análise
        future_to_symbol = {
            executor.submit(utils.analyze_symbol_for_trade, symbol, timeframe, current_params): symbol 
            for symbol in symbols_list
        }
        
        for future in as_completed(future_to_symbol):
            try:
                data = future.result()
                results.append(data)
            except Exception as exc:
                logger.error(f'{VERMELHO}Símbolo {future_to_symbol[future]} gerou uma exceção: {exc}{RESET}')

    return results
# ==================== FUNÇÃO MODIFICADA DE AVALIAÇÃO ====================

def avaliar_ativo(symbol, params, side): # <--- DEFINIÇÃO CORRETA (3 ARGUMENTOS)
    sinal_completo = False 
    status = "SEM_SINAL"
    resultado_data = utils.prepare_data_for_scan(symbol, params, lookback_days=300)
    if resultado_data is None:
        return False, {"STATUS": f"{VERMELHO}Dados Ausentes{RESET}", "ATR": 0.0}

    df, current_atr = resultado_data

    # Chama a checagem de sinal unificada (check_trade_signal está em utils.py)
    sinal_ok, detalhes = utils.check_trade_signal(df, params, side) 

    # Adiciona o lado e o status de execução para o relatório final
    detalhes['SIDE'] = side
    detalhes['SINAL_COMPLETO'] = sinal_ok
    detalhes['MOTIVO'] = f"APROVADO {side}" if sinal_ok else f"Falha no filtro ({side})"
    detalhes_rsi = {"STATUS": "N/A", "VALOR": 0.0}
    detalhes_mom = {"STATUS": "N/A", "VALOR": 0.0} 
    detalhes_ema = {"STATUS": "N/A"} 
    detalhes_adx = {"STATUS": "N/A", "VALOR": 0.0}
    return sinal_completo, {
    "STATUS": status,
    "ATR": current_atr,
    "RSI": detalhes_rsi,  # Exemplo de outro detalhe que você deve ter
    "MOMENTUM": detalhes_mom # Exemplo de outro detalhe que você deve ter
}

def load_bear_params():
    """Tenta carregar os parâmetros BEAR otimizados ou usa o default."""
    try:
        # Tenta carregar o JSON otimizado 'params_bear.json'
        with open(config.PARAMS_BEAR, "r", encoding="utf-8") as f:
            bear_params = json.load(f)
            if 'side' not in bear_params:
                 bear_params['side'] = "VENDA"
            return bear_params
    except Exception:
        # Retorna o default de Venda se falhar
        return config.DEFAULT_PARAMS_BEAR

def gerar_relatorio_horario():
    """Função placeholder que deveria gerar e salvar um relatório a cada hora."""
    
    current_hour = datetime.datetime.now().hour
    logger.info(f"{AZUL}Rotina de Relatório Horário finalizada.{RESET}")
    pass

def ciclo_principal():
    """
    Executa o scan paralelo, gerenciamento de risco e execução com SL/TP garantido.
    """
    global CB_ACTIVE, CURRENT_PARAMS

    positions = mt5.positions_get()
    
    # === CORREÇÃO DE SEGURANÇA: APLICA SL/TP EM POSIÇÕES "NAKED" (SEM SL) ===
    if positions:
        for pos in positions:
            # Verifica se a posição está sem SL (ou SL muito próximo de zero)
            if pos.sl < 0.01:
                logger.warning(f"{AMARELO}DETECTADO {pos.symbol} SEM SL (Ticket: {pos.ticket}). Tentando corrigir...{RESET}")
                
                # 1. Tenta pegar o ATR atual para calcular a distância correta
                sl_mult = CURRENT_PARAMS.get("sl_atr_mult", 2.0)
                tp_mult = CURRENT_PARAMS.get("tp_atr_mult", 4.0)
                
                # Tenta calcular ATR rápido (ou usa fallback de 1% do preço)
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

                # 2. Define Preços de SL e TP baseados no LADO da posição aberta
                if pos.type == mt5.POSITION_TYPE_BUY: # Compra
                    new_sl = pos.price_current - (atr_atual * sl_mult)
                    new_tp = pos.price_current + (atr_atual * tp_mult)
                else: # Venda
                    new_sl = pos.price_current + (atr_atual * sl_mult)
                    new_tp = pos.price_current - (atr_atual * tp_mult)

                # 3. Envia Requisição de MODIFICAÇÃO (TRADE_ACTION_SLTP)
                request = {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "symbol": pos.symbol,
                    "position": pos.ticket, 
                    "sl": float(new_sl),
                    "tp": float(new_tp)
                }
                
                res = mt5.order_send(request)
                if res.retcode == mt5.TRADE_RETCODE_DONE:
                    logger.info(f"{VERDE}SUCESSO: SL/TP Adicionado em {pos.symbol}! SL: {new_sl:.2f}, TP: {new_tp:.2f}{RESET}")
                else:
                    logger.error(f"{VERMELHO}FALHA ao corrigir {pos.symbol}: {res.comment}{RESET}")
                
                time.sleep(0.5) 

    MAX_LOSS_PERCENT = 6.0 # Define a perda máxima aceitável (Ajuste este valor)
    
    for pos in positions:
        
        # O cálculo do PnL em percentual funciona igualmente para COMPRA e VENDA
        # Retorna um valor negativo se for prejuízo
        pnl_percent = (pos.profit / (pos.price_open * pos.volume)) * 100 
        
        # Se a perda for maior que o limite (e.g., -8% é menor que -7%)
        if pnl_percent < -MAX_LOSS_PERCENT:
            
            # Puxa o tick atual para o preço de fechamento
            tick = mt5.symbol_info_tick(pos.symbol)
            if not tick: continue # Pula se não tiver tick
            
            
            # --- LÓGICA DE FECHAMENTO (COMPRA) ---
            if pos.type == mt5.POSITION_TYPE_BUY:
                action_type = mt5.ORDER_TYPE_SELL  # Vende para zerar a Compra
                price_close = tick.bid             # Venda no BID
                action_str = "VENDA (Zeragem)"
                
            # --- LÓGICA DE FECHAMENTO (VENDA) ---
            elif pos.type == mt5.POSITION_TYPE_SELL:
                action_type = mt5.ORDER_TYPE_BUY   # Compra para zerar a Venda
                price_close = tick.ask             # Compra no ASK
                action_str = "COMPRA (Zeragem)"
                
            else:
                continue # Pula se for tipo desconhecido

            
            logger.critical(f"{VERMELHO}STOP MÁXIMO ATINGIDO: {pos.symbol} com {pnl_percent:.2f}% de perda. Zerando posição ({action_str})...{RESET}")
            
            # Envia a ordem de zeragem
            request_deal = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": pos.symbol,
                "volume": pos.volume,
                "type": action_type, # <-- Varia entre BUY e SELL
                "position": pos.ticket,
                "price": price_close, # <-- Varia entre BID e ASK
                "deviation": 30,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            res = mt5.order_send(request_deal)
            
            if res.retcode == mt5.TRADE_RETCODE_DONE:
                logger.critical(f"{VERMELHO}SUCESSO: {pos.symbol} zerada por Stop Máximo. PnL: R$ {pos.profit:,.2f}{RESET}")
            else:
                logger.error(f"{VERMELHO}FALHA ao zerar {pos.symbol} por Stop Máximo: {res.comment}{RESET}")
                
            time.sleep(1)

    # 1. CARREGAR PARÂMETROS ADAPTATIVOS
    regime_str, px_ibov, ma_ibov = load_adaptive_params()
    bull_params = CURRENT_PARAMS
    bear_params = utils.load_bear_params()
    
    # 2. RELATÓRIOS E DISPLAY
    # A. Relatório de Posições (Agora com SL/TP/Distância)
    positions = mt5.positions_get() # Busca posições novamente
    print(utils.generate_positions_report(positions))
    
    print(utils.display_summary())
    print(display_optimized_params())
    print(analisar_carteira_detalhada())
    # O ERRO FOI REMOVIDO AQUI: a linha print(utils.get_position_exit_status()) não existe mais.

    # 3. CIRCUIT BREAKER (SOFT STOP)
    if CB_ACTIVE:
        logger.critical(f"{VERMELHO}CIRCUIT BREAKER ATIVO — NENHUMA NOVA ORDEM SERÁ EXECUTADA{RESET}")
        return

    # 4. PREPARAÇÃO E EXECUÇÃO DO SCANNER (para o relatório)
    ativos_para_scan = utils.get_ativos_liquidos(config.MIN_ADV_20D_BRL)
    
    # 5. Executa o scanner em paralelo para COMPRA (usa bull_params como base)
    logger.info(f"{AZUL}Iniciando varredura em {len(ativos_para_scan)} ativos...{RESET}")
    scanner_results = scanner_paralelo(ativos_para_scan, bull_params, config.TIMEFRAME_MT5) 

    # 6. Geração e Impressão do Relatório do Scanner (Top 20 e Motivos)
    scanner_report = utils.generate_scanner_report(scanner_results)
    print(scanner_report)
    
    # 7. FILTRAGEM DOS SINAIS APROVADOS PARA EXECUÇÃO (Baseado na lógica antiga)
    
    # Agora, filtraremos os resultados do scanner para execução, usando as flags can_buy/can_sell
    resultados_execucao = []
    
    for res in scanner_results:
        # Se for um sinal APROVADO, adiciona à lista para execução
        if res.get('can_buy'):
            resultados_execucao.append((res['symbol'], "COMPRA", res))
        if res.get('can_sell'):
            resultados_execucao.append((res['symbol'], "VENDA", res))
            
    if not resultados_execucao:
        logger.info(f"{AMARELO}Nenhum sinal aprovado (BUY/SELL READY) no momento. Aguardando...{RESET}")
        return

    logger.info(f"{VERDE}SINAIS APROVADOS PARA EXECUÇÃO: {len(resultados_execucao)} → Iniciando execução...{RESET}")


    # 8. EXECUÇÃO DAS ORDENS
    for symbol, side, detalhes in resultados_execucao:
        
        # === PREÇO ATUAL ===
        tick = mt5.symbol_info_tick(symbol)
        if not tick or tick.last == 0:
            logger.warning(f"Tentativa falhou → {symbol}: sem tick")
            continue
        price_current = tick.bid if side == "COMPRA" else tick.ask

        # === ATR E PARÂMETROS ===
        # Nota: A função analyze_symbol_for_trade não calcula ATR, então usaremos um valor padrão
        # Para que o SL funcione, você deve integrar o cálculo do ATR na sua função 'avaliar_ativo' ou usá-lo aqui.
        # Assumindo ATR de 1% do preço para continuar a execução.

        # --- CÁLCULO ATR: (Você deve ter uma função para isso, mas vou simular o cálculo simples) ---
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

        sl_mult = CURRENT_PARAMS.get("sl_atr_mult", 2.0)
        tp_mult = CURRENT_PARAMS.get("tp_atr_mult", 4.0)

        # === CÁLCULO DE SL E TP ===
        if side == "COMPRA":
            sl_price = price_current - (current_atr * sl_mult)
            tp_price = price_current + (current_atr * tp_mult)
        else:
            sl_price = price_current + (current_atr * sl_mult)
            tp_price = price_current - (current_atr * tp_mult)

        sl_price = max(sl_price, 0.01)
        tp_price = max(tp_price, 0.01)

        # === CÁLCULO DO LOTE (1% DO EQUITY) ===
        acc = mt5.account_info()
        if not acc or acc.equity <= 0:
            logger.error("Não foi possível obter equity da conta")
            continue

        risco_reais = acc.equity * 0.01  # 1% por trade
        distancia_sl = abs(price_current - sl_price)
        if distancia_sl < 0.01:
            continue

        lote_bruto = risco_reais / (distancia_sl * 100)  # 100 contratos = 1 ponto
        lote = int(math.floor(lote_bruto / 100) * 100)
        lote = max(100, lote)  # mínimo 100 contratos

        if lote < 100:
            logger.warning(f"{symbol} → Lote calculado muito baixo ({lote}). Pulando.")
            continue

        # === EXECUÇÃO COM SL/TP GARANTIDO ===
        logger.info(f"{AZUL}EXECUTANDO → {symbol} {side} | Lote: {lote:,} | Preço: {price_current:.2f} | SL: {sl_price:.2f} | TP: {tp_price:.2f}{RESET}")

        result = execute_iceberg_order(
            symbol=symbol,
            side=side,
            lote_calc=lote,
            price_current=price_current,
            sl_price_simulado=sl_price,
            tp_price_simulado=tp_price
        )

        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"{VERDE}SUCESSO TOTAL → {symbol} {side} {lote:,} lotes | Ticket: {result.order} | SL/TP COLOCADOS{RESET}")
        else:
            logger.error(f"{VERMELHO}Falha ao executar {symbol} {side}{RESET}")

    logger.info(f"{AZUL}Ciclo concluído. Próxima varredura em {config.CHECK_INTERVAL_SLOW}s...{RESET}")

    # 7. GERAÇÃO DE RELATÓRIO HORÁRIO (se a hora mudar)
    gerar_relatorio_horario()
    
    # ==================== FUNÇÃO MAIN (LOOP) ====================

def main():
    global CB_ACTIVE, LAST_CB_CHECK_DAY, DAILY_START_EQUITY
    
    # 1. TENTA CONEXÃO SILENCIOSA COM MT5 ABERTO
    if not mt5.initialize():
        logger.warning("Falha na conexão silenciosa. Tentando inicialização completa...")
        
        # 2. TENTA CONEXÃO COMPLETA (LOGIN E SENHA)
        if not mt5.initialize(login=config.LOGIN, password=config.PASSWORD, server=config.SERVER):
            logger.critical(f"Falha na inicialização do MT5 (Login/Senha). Erro: {mt5.last_error()}. Encerrando.")
            time.sleep(10)
            return 
    
    logger.info(f"{VERDE}Conexão MT5 Estabelecida com sucesso!{RESET}")
    
    # Inicia o Loop Principal
    while True:
        try:
            current_time = datetime.datetime.now().time()
            acc_info = mt5.account_info()
            
            # 1. Reset Diário (Início do Dia)
            if datetime.datetime.now().day != LAST_CB_CHECK_DAY:
                CB_ACTIVE = False # Libera o CB no novo dia
                LAST_CB_CHECK_DAY = datetime.datetime.now().day
                DAILY_START_EQUITY = 0.0
                logger.info(f"{AZUL}--- RESET DIÁRIO CONCLUÍDO ---{RESET}")

            # Define Equity Inicial
            if DAILY_START_EQUITY == 0.0:
                 DAILY_START_EQUITY = acc_info.equity
                 logger.info(f"{AZUL}📈 Equity Inicial do Dia Definido: R$ {DAILY_START_EQUITY:,.2f}{RESET}")
                
            # 2. Checagem do Circuit Breaker (Soft Stop Unificado)
            tick_data = mt5.symbol_info_tick(config.IBOV_SYMBOL)
            
            # Comente a linha abaixo para produção e use a checagem real:
            # if utils.check_circuit_breakers(acc_info, tick_data, DAILY_START_EQUITY):
            #     CB_ACTIVE = True 
            CB_ACTIVE = False # DEBUG: Força a flag para False para garantir o teste do scanner

            # 3. Verifica Horário de Operação
                        # 3. Verifica Horário de Operação → MODO TESTE 24H (COMENTE OU DESATIVE PARA PRODUÇÃO)
            FORCAR_OPERACAO_24H = False  # <=== MUDE PARA False QUANDO FOR PRA CONTA REAL!!!

            if not FORCAR_OPERACAO_24H:
                if not (config.START_TIME <= current_time <= config.END_TIME):
                    print(f"FORA DE HORÁRIO (teste 24h desativado): {current_time.strftime('%H:%M:%S')} | Equity: R$ {acc_info.equity:,.2f}", end='\r')
                    time.sleep(60)
                    continue
            else:
                print(f"{AZUL}MODO TESTE 24H ATIVO → Operando fora do horário normal{RESET}", end='\r')
                
            # 4. Executa o Ciclo Principal (Display e Execução)
            ciclo_principal() 
            
            time.sleep(config.CHECK_INTERVAL_SLOW)
            
        except Exception as e:
            logger.error(f"{VERMELHO}Exceção não tratada no Loop Principal: {e}{RESET}")
            time.sleep(config.CHECK_INTERVAL_SLOW)

if __name__ == "__main__":
    main()