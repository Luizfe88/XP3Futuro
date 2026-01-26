# bot.py – EXECUTOR INSTITUCIONAL B3 (V3) - NOVO DISPLAY/LOOP

import MetaTrader5 as mt5
import time
import os
import json
import threading
import random
import datetime 
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import numpy as np

import config
import utils
from utils import logger, Fore, Style, VERDE, AMARELO, VERMELHO, AZUL, ROXO, RESET

# ==================== VARIÁVEIS GLOBAIS DE CONTROLE ====================
CURRENT_PARAMS = config.DEFAULT_PARAMS
SLIPPAGE_HISTORY = [] 

# NOVOS: Flag Global para Circuit Breaker e rastreio diário
CB_ACTIVE = False # Se True, impede novas compras/vendas
LAST_CB_CHECK_DAY = datetime.datetime.now().day # Controla o reset diário

# ==================== FUNÇÕES AUXILIARES ====================

def load_adaptive_params():
    """Carrega parâmetros baseados no Regime de Mercado E aplica Walk-Forward."""
    global CURRENT_PARAMS
    regime_str, px_ibov, ma_ibov, vix_br = utils.get_market_regime()
    # get_market_regime retorna: regime_str, IBOV_Price, IBOV_MA200, VIX_BR
    
    file_map = {
        "STRONG_BULL": config.PARAMS_STRONG_BULL,
        "BULL": config.PARAMS_BULL,
        "SIDEWAYS": config.PARAMS_SIDEWAYS,
        "BEAR": config.PARAMS_BEAR,
        "CRISIS": config.PARAMS_CRISIS
    }
    
    try:
        filename = file_map.get(regime_str, config.PARAMS_SIDEWAYS)
        with open(filename, 'r') as f:
            params = json.load(f)
            CURRENT_PARAMS = params
            logger.info(f"{AZUL}Parâmetros Adaptativos Carregados: Regime {regime_str} de {filename}{RESET}")
    except Exception as e:
        logger.error(f"{VERMELHO}Falha ao carregar parâmetros do regime {regime_str}. Usando default. Erro: {e}{RESET}")
        CURRENT_PARAMS = config.DEFAULT_PARAMS
        
    return regime_str, px_ibov, ma_ibov

def display_optimized_params():
    """Formata os parâmetros atuais para exibição."""
    p = CURRENT_PARAMS
    output = f"\n{AZUL}=== PARÂMETROS ADAPTATIVOS ({p.get('regime', 'N/A').upper()}) ==={RESET}\n"
    output += f"{'SIDE (OPERAÇÃO)':<30}: {p.get('side', 'N/A')}\n"
    output += f"{'EMA RÁPIDA / LENTA':<30}: {p.get('ema_fast', '-')}/{p.get('ema_slow', '-')}\n"
    output += f"{'RSI NÍVEL':<30}: >{p.get('rsi_level', '-')}\n"
    output += f"{'MOMENTUM MÍNIMO':<30}: >{p.get('momentum_min', '-')}\n"
    output += f"{'ADX MÍNIMO':<30}: >{p.get('adx_min', '-')}\n"
    output += f"{'SHARPE OTIMIZADO (PROXY)':<30}: {p.get('score', '-'):.2f}\n"
    output += f"{'STOP LOSS (ATR Mult)':<30}: {p.get('sl_atr_mult', '-')}\n"
    output += f"{'TAKE PROFIT (ATR Mult)':<30}: {p.get('tp_mult', '-')}\n"
    output += "-" * 110
    return output

def analisar_carteira_detalhada():
    """Obtém e formata informações detalhadas das posições abertas."""
    positions = mt5.positions_get()
    output = f"\n{AZUL}=== RELATÓRIO DA CARTEIRA ({len(positions) if positions else 0} POSIÇÕES) ==={RESET}\n"
    output += f"{'SÍMBOLO':<10}{'LOTE':<8}{'PREÇO MÉDIO':<15}{'PREÇO ATUAL':<15}{'PNL (R$)':<15}{'PNL (%)':<10}{'SIDE':<8}\n"
    output += "-" * 81 + "\n"
    
    total_pnl = 0.0
    if positions:
        for pos in positions:
            symbol = pos.symbol
            lote = pos.volume
            pm = pos.price_open
            pc = pos.price_current
            pnl_reais = pos.profit
            
            # PNL em porcentagem do valor da posição
            pnl_pct = (pc / pm - 1) * 100 if pos.type == mt5.POSITION_TYPE_BUY else (pm / pc - 1) * 100
            
            pnl_color = VERDE if pnl_reais >= 0 else VERMELHO
            side = "COMPRA" if pos.type == mt5.POSITION_TYPE_BUY else "VENDA"
            
            output += f"{symbol:<10}{lote:<8.0f}{pm:<15.2f}{pc:<15.2f}{pnl_color}{pnl_reais:<15.2f}{RESET}{pnl_color}{pnl_pct:<10.2f}{RESET}{side:<8}\n"
            total_pnl += pnl_reais
            
    output += "-" * 81 + "\n"
    output += f"{'TOTAL PNL FLUTUANTE':<50}{VERDE if total_pnl >= 0 else VERMELHO}{total_pnl:,.2f} R${RESET}\n"
    output += "---" * 27
    return output

def execute_iceberg_order(symbol, type_operation, volume, price_limit, deviation=20):
    """Simula a execução de uma ordem Iceberg (envio de ordem MT5)."""
    # Lógica de envio de ordem (omitida por brevidade)
    logger.info(f"{AZUL}Execução: Tentando enviar ordem {type_operation} {volume} lotes em {symbol} @ {price_limit}{RESET}")
    return True

def avaliar_ativo(symbol, regime_str):
    """Aplica a regra otimizada ao ativo e checa se deve ser negociado."""
    
    # 1. Filtro de Liquidez
    if not utils.is_liquid_asset(symbol):
        return False, f"Falha Liquidez: {symbol}"
        
    # 2. Parâmetros e Dados
    params = CURRENT_PARAMS
    # Obtém os dados mais recentes (M5 ou D1, dependendo da configuração)
    rates = mt5.copy_rates_from_pos(symbol, config.TIMEFRAME_MT5, 1, 250)
    if rates is None or len(rates) < 60: return False, f"Dados insuficientes: {symbol}"
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    
    # 3. Análise Técnica (Omitida por brevidade, mas deve usar os params de CURRENT_PARAMS)
    
    # 4. Lógica de Sinais (Simplificada)
    side_op = params.get('side')
    
    # Retorna True se a condição for atendida.
    # Esta lógica deve ser refinada para refletir a estratégia real de trading.
    if side_op == "COMPRA" and random.random() < 0.05: # 5% de chance de comprar
         return True, f"Sinal de {side_op}"
    if side_op == "VENDA" and random.random() < 0.05: # 5% de chance de vender
         return True, f"Sinal de {side_op}"
         
    return False, "Sem Sinal"

# ==================== CICLO PRINCIPAL (Controle do CB) ====================

def ciclo_principal():
    """
    Função principal que faz o loop, o scan, a execução e o display no console.
    """
    global CB_ACTIVE
    
    # 1. LIMPA A TELA & CABEÇALHO
    os.system('cls' if os.name == 'nt' else 'clear') 
    print(f"{AZUL}=== BOT ELITE 2026 PRO (INSTITUCIONAL KERNEL) ==={RESET}")
    
    # 3. REGIME DE MERCADO (Com Fallback de Segurança)
    regime_str = "SIDEWAYS" # FALLBACK: Garante que a variável exista
    px_ibov = 0.0
    ma_ibov = 0.0
    
    try:
        regime_str, px_ibov, ma_ibov = load_adaptive_params()
    except Exception as e:
        # Se a carga falhar, as variáveis mantêm os valores de fallback
        logger.error(f"{VERMELHO}Falha crítica ao carregar parâmetros adaptativos. Usando {regime_str} (Default). Erro: {e}{RESET}")
    
    # 4. RESUMO FINANCEIRO
    account_info = mt5.account_info()
    
    # CORRIGIDO: O 'if' e o seu bloco devem ter apenas 4 espaços de indentação.
    if account_info is None:
        logger.error(f"{VERMELHO}Falha ao obter informações da conta MT5.{RESET}")
        return
    
    utils.EQUITY_DROP_HISTORY.append({'time': datetime.datetime.now(), 'equity': account_info.equity})

    # Obtém o PNL
    pl_reais, pl_pct = utils.get_daily_profit_loss()
    
    print(f"\n{AZUL}=== RESUMO FINANCEIRO (Conta: {account_info.login}) ==={RESET}")
    print("-" * 110)
    
    pl_color = VERDE if pl_reais >= 0 else VERMELHO
    
    print(f"{'CAPITAL TOTAL (EQUITY)':<30}: R$ {account_info.equity:,.2f}")
    print(f"{'DINHEIRO DISPONÍVEL (LIVRE)':<30}: R$ {account_info.margin_free:,.2f}")
    print(f"{'INVESTIDO (MARGEM EM USO)':<30}: R$ {account_info.margin:,.2f}")
    
    print(f"{'PNL FLUTUANTE DIÁRIO':<30}: {pl_color}R$ {pl_reais:,.2f} ({pl_pct:+.2f}%){RESET}")
    print("-" * 110)

    # 5. PARÂMETROS OTIMIZADOS
    print(display_optimized_params())
    
    # 6. RELATÓRIO DA CARTEIRA
    print(analisar_carteira_detalhada())
    
    
def get_daily_performance(symbol):
    """
    Calcula a performance diária de um símbolo usando o preço de fechamento do D-1 (dia anterior).
    Usada em ThreadPoolExecutor para agilizar a busca de dados.
    """
    # Busca a última barra D1 fechada (D-1)
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_D1, 1, 1) 
    
    if rates and len(rates) == 1:
        close_d1 = rates[0]['close']
        tick = mt5.symbol_info_tick(symbol)
        
        if tick and close_d1 > 0:
            current_price = tick.last
            # (Preço Atual / Fechamento D-1) - 1
            performance = (current_price / close_d1 - 1) * 100
            return performance
            
    return None

def display_top_10_performers():
    """
    Calcula e exibe as 10 melhores ações do universo B3 em termos de variação diária.
    """
    symbols = list(config.UNIVERSE_B3.keys())
    performance_list = []
    
    # 1. Paraleliza a coleta de dados de performance
    with ThreadPoolExecutor(max_workers=os.cpu_count() * 2) as executor:
        future_to_symbol = {executor.submit(get_daily_performance, symbol): symbol for symbol in symbols}
        
        for future in future_to_symbol:
            symbol = future_to_symbol[future]
            try:
                perf_pct = future.result()
                if perf_pct is not None:
                    performance_list.append({'symbol': symbol, 'performance': perf_pct})
            except Exception as e:
                # Silencia erros de dados insuficientes ou falha de tick
                pass 
                
    # 2. Ordena (Decrescente) e seleciona o Top 10
    performance_list.sort(key=lambda x: x['performance'], reverse=True)
    top_10 = performance_list[:10]
    
    # 3. Formata a Saída
    output = f"\n{AZUL}=== TOP 10 PERFORMERS HOJE ({datetime.datetime.now().strftime('%H:%M')}) ==={RESET}\n"
    output += f"{'RANK':<6}{'SÍMBOLO':<10}{'VAR. (%)':<10}\n"
    output += "-" * 26 + "\n"
    
    for i, item in enumerate(top_10):
        perf = item['performance']
        color = VERDE if perf >= 0 else VERMELHO
        output += f"{i+1:<6}{item['symbol']:<10}{color}{perf:+.2f}{RESET}\n"
        
    output += "---" * 9
    return output

# ==================== CICLO PRINCIPAL (Controle do CB) ====================

def ciclo_principal():
    """
    Função principal que faz o loop, o scan, a execução e o display no console.
    """
    global CB_ACTIVE
    
    # 1. LIMPA A TELA & CABEÇALHO
    os.system('cls' if os.name == 'nt' else 'clear') 
    print(f"{AZUL}=== BOT ELITE 2026 PRO (INSTITUCIONAL KERNEL) ==={RESET}")
    
    # =======================================================
    # 3. REGIME DE MERCADO (Com Fallback de Segurança)
    # CORREÇÃO CRÍTICA: Define fallback ANTES do try para garantir o escopo!
    # =======================================================
    regime_str = "SIDEWAYS" # FALLBACK GARANTIDO. O robô usará "SIDEWAYS" se a carga falhar.
    px_ibov = 0.0
    ma_ibov = 0.0
    
    try:
        # AQUI tentamos carregar os valores otimizados
        regime_str, px_ibov, ma_ibov = load_adaptive_params()
    except Exception as e:
        # Se a carga falhar, as variáveis mantêm os valores de fallback
        logger.error(f"{VERMELHO}Falha crítica ao carregar parâmetros adaptativos. Usando {regime_str} (Default). Erro: {e}{RESET}")
    
    # 4. RESUMO FINANCEIRO
    account_info = mt5.account_info()
    
    if account_info is None:
        logger.error(f"{VERMELHO}Falha ao obter informações da conta MT5.{RESET}")
        return
    
    utils.EQUITY_DROP_HISTORY.append({'time': datetime.datetime.now(), 'equity': account_info.equity})

    # Obtém o PNL
    pl_reais, pl_pct = utils.get_daily_profit_loss()
    
    print(f"\n{AZUL}=== RESUMO FINANCEIRO (Conta: {account_info.login}) ==={RESET}")
    print("-" * 110)
    
    pl_color = VERDE if pl_reais >= 0 else VERMELHO
    
    print(f"{'CAPITAL TOTAL (EQUITY)':<30}: R$ {account_info.equity:,.2f}")
    print(f"{'DINHEIRO DISPONÍVEL (LIVRE)':<30}: R$ {account_info.margin_free:,.2f}")
    print(f"{'INVESTIDO (MARGEM EM USO)':<30}: R$ {account_info.margin:,.2f}")
    
    print(f"{'PNL FLUTUANTE DIÁRIO':<30}: {pl_color}R$ {pl_reais:,.2f} ({pl_pct:+.2f}%){RESET}")
    print("-" * 110)

    # 5. PARÂMETROS OTIMIZADOS
    print(display_optimized_params())
    
    # 6. RELATÓRIO DA CARTEIRA
    print(analisar_carteira_detalhada())
    
    # DISPLAY TOP 10 PERFORMERS
    print(display_top_10_performers())
    
    # Condição de Circuit Breaker
    if CB_ACTIVE:
        print(f"\n{VERMELHO}--- 🚫 CIRCUIT BREAKER ATIVADO (SOFT STOP) ---{RESET}")
        print(f"{VERMELHO}⚠️ ALERTA DE RISCO: Limite de VaR diário ({config.VAR_95_DAILY_LIMIT*100:.1f}%) atingido.{RESET}")
        print(f"{VERMELHO}Operações de COMPRA/VENDA (Scan e Execução) estão SUSPENSAS. Monitoramento Ativo.{RESET}")
        print("=" * 110)
        return 
        
    # 7. SCAN DE OPORTUNIDADES (Só roda se CB_ACTIVE for False)
    
    ativos_para_scan = list(config.UNIVERSE_B3.keys())
    random.shuffle(ativos_para_scan) 
    
    logger.info(f"{ROXO}Iniciando Scan de Oportunidades em {len(ativos_para_scan)} ativos...{RESET}")
    
    resultados = []
    # AQUI regime_str está garantido e pode ser usado.
    with ThreadPoolExecutor(max_workers=os.cpu_count() * 2) as executor:
        future_to_symbol = {executor.submit(avaliar_ativo, symbol, regime_str): symbol for symbol in ativos_para_scan}
    
    for future in future_to_symbol:
        pode_negociar, log = future.result()
        if pode_negociar:
            resultados.append((future_to_symbol[future], log))
        else:
             pass

    if resultados:
        logger.info(f"{VERDE}✅ {len(resultados)} Oportunidade(s) Encontrada(s). Iniciando Execução.{RESET}")
        
        # 8. EXECUÇÃO DE ORDENS
        for symbol, log in resultados:
            logger.info(f"{AMARELO}ORDEM: {symbol} | {log}{RESET}")
            lote_calc = utils.calcular_tamanho_posicao(symbol, 0, True)
            
            if lote_calc > 0:
                execute_iceberg_order(symbol, CURRENT_PARAMS.get('side'), lote_calc, mt5.symbol_info_tick(symbol).last)

    else:
        logger.info("Nenhuma oportunidade encontrada no ciclo atual.")

def bot_loop_wrapper():
    """Loop principal que gerencia o tempo, a conexão e o Circuit Breaker."""
    global CB_ACTIVE, LAST_CB_CHECK_DAY
    
    logger.info(f"{VERDE}Bot Iniciado - Modo Institucional (Kernel V3){RESET}")
    
    # 1. Inicia Trailing em Thread Separada
    t_trail = threading.Thread(target=utils.trailing_stop_service, daemon=True)
    logger.info(f"{AZUL}Serviço de Trailing Stop iniciado (Thread).{RESET}")
    t_trail.start()
    
    while True:
        try:
            current_day = datetime.datetime.now().day
            current_time = datetime.datetime.now().time()
            
            if os.path.exists("STOP.CMD"): 
                logger.warning("Arquivo STOP.CMD encontrado. Encerrando bot.")
                break
                
            acc_info = mt5.account_info()
            if acc_info is None:
                logger.error("Falha ao obter account_info. Tentando reconexão.")
                utils.check_mt5_connection()
                time.sleep(config.CHECK_INTERVAL_SLOW)
                continue
            
            # --- NOVO: Lógica de Reset Diário do Circuit Breaker ---
            # Se o dia mudou E estamos dentro do horário de operação: Reseta a flag
            if current_day != LAST_CB_CHECK_DAY and (config.START_TIME <= current_time <= config.END_TIME):
                CB_ACTIVE = False # <--- AQUI O ROBÔ VOLTA A OPERAR
                LAST_CB_CHECK_DAY = current_day
                logger.info(f"{AZUL}🔄 Circuit Breaker Reset: Novo dia de negociação. Operações reativadas.{RESET}")
            # --------------------------------------------------------
                
            # 2. Checagem do Circuit Breaker
            tick_data = mt5.symbol_info_tick(config.IBOV_SYMBOL)
            
            # Se utils.check_circuit_breakers retornar True, a flag é ativada.
            if utils.check_circuit_breakers(acc_info, tick_data):
                CB_ACTIVE = True 

            # 3. Verifica Horário de Operação
            if not (config.START_TIME <= current_time <= config.END_TIME):
                print(f"FORA DE HORÁRIO: {current_time.strftime('%H:%M:%S')} | Equity: R$ {acc_info.equity:,.2f}", end='\r')
                time.sleep(60)
                continue
                
            # 4. Executa o Ciclo Principal (Display e Execução)
            # O ciclo_principal verifica a flag CB_ACTIVE antes de executar novas ordens.
            ciclo_principal() 
            
        except Exception as e:
            logger.error(f"{VERMELHO}Erro no loop principal: {e}{RESET}")

        time.sleep(config.CHECK_INTERVAL_SLOW)

if __name__ == '__main__':
    if utils.check_mt5_connection():
        bot_loop_wrapper()