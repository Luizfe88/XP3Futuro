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
from utils import logger, Fore, Style, VERDE, AMARELO, VERMELHO, AZUL, ROXO, RESET

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

# --- Fim da função substituída ---

def execute_iceberg_order(symbol, side, lote_calc, price_current):
    """
    FUNÇÃO DE EXECUÇÃO REAL DA ORDEM NO MT5.
    Envia uma ordem de compra ou venda a mercado (MKT).
    """
    
    if side in ["COMPRA", "BUY"]:
        action_type = mt5.ORDER_TYPE_BUY
        order_side_name = "COMPRA"
    elif side in ["VENDA", "SELL"]:
        action_type = mt5.ORDER_TYPE_SELL
        order_side_name = "VENDA"
    else:
        # Este log só deve ser visto se houver erro no fallback
        logger.error(f"{VERMELHO}SIDE inválido '{side}' para execução em {symbol}.{RESET}")
        return None

    # O volume (lote) deve ser arredondado para baixo. Lote mínimo de 100 para a maioria.
    volume_final = math.floor(lote_calc)
    
    if volume_final < 100: 
        logger.warning(f"Ordem de {symbol} cancelada: Lote calculado ({lote_calc:.2f}) é menor que 100.")
        return None
        
    # --- Requisição de Ordem a Mercado (Baseado no exemplo do crazy.py) ---
    request = {
        "action": mt5.TRADE_ACTION_DEAL,      
        "symbol": symbol,                     
        "volume": float(volume_final),        
        "type": action_type,                  
        "price": price_current,               
        "deviation": 50,                      
        "type_filling": mt5.ORDER_FILLING_RETURN, 
        "type_time": mt5.ORDER_TIME_GTC,      
        "comment": f"BOT ELITE {order_side_name}",
    }

    # ENVIA A ORDEM REAL!
    result = mt5.order_send(request)
    
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logger.error(f"{VERMELHO}❌ FALHA na execução {symbol} ({order_side_name}): Código {result.retcode}. Comentário: {result.comment}{RESET}")
        
    else:
        logger.info(f"{VERDE}✅ SUCESSO! {symbol} | {order_side_name} {volume_final} Lotes | Ticket: {result.order}{RESET}")

    return result
# ==================== FUNÇÕES NOVAS PARA O RELATÓRIO ====================

def colorir_resultado(pass_status, valor):
    """Função auxiliar para aplicar as cores VERDE/VERMELHO."""
    cor = VERDE if pass_status else VERMELHO
    return f"{cor}{valor}{RESET}"

def display_signal_breakdown(resultados_scan, top_n=10):
   relatorio = utils.scanner_full_report(resultados_scan_detalhado, top_n)
   print(relatorio)


# ==================== FUNÇÃO MODIFICADA DE AVALIAÇÃO ====================

def avaliar_ativo(symbol, params, side): # <--- DEFINIÇÃO CORRETA (3 ARGUMENTOS)
    """
    Função de checagem de sinal unificada.
    """
    
    # CORREÇÃO CRÍTICA: O parâmetro 'params' DEVE ser passado
    df = utils.prepare_data_for_scan(symbol, params) 
    
    if df is None:
        return False, {"STATUS": f"{VERMELHO}Dados Ausentes{RESET}", "EMA": "-", "RSI": "-", "ADX": "-", "MOM": "-"}
        
    # Chama a checagem de sinal unificada (check_trade_signal está em utils.py)
    sinal_ok, detalhes = utils.check_trade_signal(df, params, side) 

    # Adiciona o lado e o status de execução para o relatório final
    detalhes['SIDE'] = side
    detalhes['SINAL_COMPLETO'] = sinal_ok
    detalhes['MOTIVO'] = f"APROVADO {side}" if sinal_ok else f"Falha no filtro ({side})"

    return sinal_ok, detalhes

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


# ==================== CICLO PRINCIPAL MODIFICADO ====================

def ciclo_principal():
    """
    Função principal que faz o loop, o scan, a execução e o display no console.
    """
    global CB_ACTIVE, CURRENT_PARAMS
    
    os.system('cls' if os.name == 'nt' else 'clear') 
    print(f"{AZUL}=== BOT ELITE 2026 PRO (INSTITUCIONAL KERNEL) ==={RESET}")
    
    regime_str = "SIDEWAYS" 
    px_ibov = 0.0
    ma_ibov = 0.0
    
    try:
        # Carrega BULL (regime atual) e atualiza CURRENT_PARAMS
        regime_str, px_ibov, ma_ibov = load_adaptive_params() 
        bull_params = CURRENT_PARAMS
        
        # Carrega BEAR (parâmetros de Venda)
        bear_params = load_bear_params()
        
    except Exception as e:
        logger.error(f"{VERMELHO}Falha crítica ao carregar parâmetros. Erro: {e}{RESET}")
        return # Falha crítica, sai do ciclo
    
    account_info = mt5.account_info()
    
    if account_info is None:
        logger.error(f"{VERMELHO}Falha ao obter informações da conta MT5.{RESET}")
        return
    
    utils.EQUITY_DROP_HISTORY.append({'time': datetime.datetime.now(), 'equity': account_info.equity})

    pl_reais, pl_pct = utils.get_daily_profit_loss() 
    
    print(f"\n{AZUL}=== RESUMO FINANCEIRO (Conta: {account_info.login}) ==={RESET}")
    print("-" * 110)
    
    pl_color = VERDE if pl_reais >= 0 else VERMELHO
    
    print(f"{'CAPITAL TOTAL (EQUITY)':<30}: R$ {account_info.equity:,.2f}")
    print(f"{'DINHEIRO DISPONÍVEL (LIVRE)':<30}: R$ {account_info.margin_free:,.2f}")
    print(f"{'INVESTIDO (MARGEM EM USO)':<30}: R$ {account_info.margin:,.2f}")
    
    print(f"{'PNL FLUTUANTE DIÁRIO':<30}: {pl_color}R$ {pl_reais:,.2f} ({pl_pct:+.2f}%){RESET}")
    
    if DAILY_START_EQUITY > 0:
        daily_drawdown_pct = (DAILY_START_EQUITY - account_info.equity) / DAILY_START_EQUITY
        dd_color = VERMELHO if daily_drawdown_pct >= config.MAX_DAILY_DRAWDOWN_PCT else AZUL
        print(f"{'EQUITY INICIAL DO DIA':<30}: R$ {DAILY_START_EQUITY:,.2f}")
        print(f"{'DD MÁXIMO ATUAL':<30}: {dd_color}{daily_drawdown_pct*100:,.2f}% (Limite: {config.MAX_DAILY_DRAWDOWN_PCT*100:.1f}%){RESET}")

    print("-" * 110)

    print(display_optimized_params())
    print(analisar_carteira_detalhada()) # <-- AGORA BUSCA DADOS REAIS DO MT5!

    # --- NOVO FLUXO: CONTENÇÃO DE RISCO (DEVE VIR APÓS O RELATÓRIO) ---
    if CB_ACTIVE:
        print(f"\n{VERMELHO}--- 🚫 CIRCUIT BREAKER ATIVADO (SOFT STOP) ---{RESET}")
        print(f"{VERMELHO}⚠️ ALERTA DE RISCO: Limite de Drawdown ou VaR Diário atingido. Scanner Desativado.{RESET}")
        print("=" * 110)
        return 
        
    # ==================== SCAN DE OPORTUNIDADES UNIFICADO ====================
    logger.info(f"{AZUL}Passou pelo CB Check. Iniciando o ThreadPoolExecutor para Scan...{RESET}")

    # GARANTIA: listas sempre existem (evita NameError)
    resultados_execucao = []
    resultados_scan_detalhado = []

    try:
        ativos_para_scan = list(config.UNIVERSE_B3.keys())
        random.shuffle(ativos_para_scan)

        tasks = []
        for symbol in ativos_para_scan:
            tasks.append((symbol, bull_params, "COMPRA"))
            tasks.append((symbol, bear_params, "VENDA"))

        logger.info(f"{ROXO}Iniciando Scan de {len(tasks)} tarefas (COMPRA + VENDA) em {len(ativos_para_scan)} ativos...{RESET}")

        with ThreadPoolExecutor(max_workers=os.cpu_count() * 2) as executor:
            future_to_task = {executor.submit(avaliar_ativo, *task): task for task in tasks}

            for future in as_completed(future_to_task):
                task = future_to_task[future]
                symbol, params, side = task
                try:
                    result_tuple = future.result()

                    if not isinstance(result_tuple, tuple) or len(result_tuple) != 2:
                        continue

                    sinal_ok, detalhes = result_tuple

                    if not isinstance(detalhes, dict):
                        continue

                    # Sempre guarda para o relatório detalhado
                    resultados_scan_detalhado.append((symbol, side, detalhes))

                    # Só executa se o sinal foi aprovado
                    if sinal_ok:
                        resultados_execucao.append((symbol, side, detalhes.get("MOTIVO", "APROVADO"), params))

                except Exception as e:
                    logger.error(f"{VERMELHO}Erro ao avaliar {symbol} ({side}): {e}{RESET}")

        # EXIBE O RELATÓRIO SEMPRE (mesmo que tenha dado algum erro no meio)
        print(utils.scanner_full_report(resultados_scan_detalhado, top_n=20))

        # EXECUÇÃO DAS ORDENS
        if resultados_execucao:
            logger.info(f"{VERDE}Encontradas {len(resultados_execucao)} oportunidade(s). Iniciando execução...{RESET}")
            for symbol, side, motivo, params_used in resultados_execucao:
                logger.info(f"{AMARELO}EXECUTANDO → {symbol} | {side} | {motivo}{RESET}")

                tick = mt5.symbol_info_tick(symbol)
                if tick is None or not tick.last:
                    continue
                price_current = tick.last

                sl_mult = params_used.get('sl_atr_mult', 2.0)
                if side == "COMPRA":
                    sl_price_simulado = price_current * (1 - sl_mult * 0.005)
                    is_buy = True
                else:
                    sl_price_simulado = price_current * (1 + sl_mult * 0.005)
                    is_buy = False

                lote_calc = utils.calcular_tamanho_posicao(symbol, sl_price_simulado, is_buy)

                if lote_calc >= 100:
                    execute_iceberg_order(symbol, side, lote_calc, price_current)
                else:
                    logger.warning(f"Lote muito pequeno ({lote_calc:.0f}) para {symbol} → ordem cancelada.")
        else:
            logger.info("Nenhuma oportunidade válida neste ciclo.")

    except Exception as e:
        logger.error(f"{VERMELHO}CRÍTICO: Falha geral no scan → {e}{RESET}")
        # Mesmo com erro total, tenta mostrar o que já conseguiu coletar
        if resultados_scan_detalhado:
            print(utils.scanner_full_report(resultados_scan_detalhado, top_n=20))

        if resultados_execucao:
            logger.info(f"{VERDE}Encontradas {len(resultados_execucao)} oportunidade(s). Iniciando execução...{RESET}")
            for symbol, side, motivo, params_used in resultados_execucao:
                logger.info(f"{AMARELO}EXECUTANDO → {symbol} | {side} | {motivo}{RESET}")

                tick = mt5.symbol_info_tick(symbol)
                if tick is None or not tick.last:
                    continue
                price_current = tick.last

                sl_mult = params_used.get('sl_atr_mult', 2.0)
                if side == "COMPRA":
                    sl_price_simulado = price_current * (1 - sl_mult * 0.005)
                    is_buy = True
                else:
                    sl_price_simulado = price_current * (1 + sl_mult * 0.005)
                    is_buy = False

                lote_calc = utils.calcular_tamanho_posicao(symbol, sl_price_simulado, is_buy)

                if lote_calc >= 100:
                    # === VERIFICA SE JÁ TEM POSIÇÃO ABERTA ===
                    positions = mt5.positions_get(symbol=symbol)
                    if positions and len(positions) > 0:
                        logger.info(f"{AMARELO}Ignorado {symbol} ({side}): já tem posição aberta.{RESET}")
                        continue
                    # ===========================================

                    execute_iceberg_order(symbol, side, lote_calc, price_current)
                else:
                    logger.warning(f"Lote insuficiente ({lote_calc:.0f}) para {symbol}")
        else:
            logger.info("Nenhuma oportunidade válida neste ciclo.")

    except Exception as e:
        logger.error(f"{VERMELHO}CRÍTICO: Falha Geral no Fluxo Principal de Scanning: {e}{RESET}")
        # logger.error(traceback.format_exc()) # Comentado

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
            if not (config.START_TIME <= current_time <= config.END_TIME):
                print(f"FORA DE HORÁRIO: {current_time.strftime('%H:%M:%S')} | Equity: R$ {acc_info.equity:,.2f}", end='\r')
                time.sleep(60)
                continue
                
            # 4. Executa o Ciclo Principal (Display e Execução)
            ciclo_principal() 
            
            time.sleep(config.CHECK_INTERVAL_SLOW)
            
        except Exception as e:
            logger.error(f"{VERMELHO}Exceção não tratada no Loop Principal: {e}{RESET}")
            time.sleep(config.CHECK_INTERVAL_SLOW)

if __name__ == "__main__":
    main()