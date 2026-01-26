# bot.py – EXECUTOR INSTITUCIONAL B3 (V3) - NOVO DISPLAY/LOOP

import MetaTrader5 as mt5
import time
import os
import json
import threading
import random
import datetime # Importado como módulo
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import numpy as np

import config
import utils
# Importação direta das cores
from utils import logger, Fore, Style, VERDE, AMARELO, VERMELHO, AZUL, ROXO, RESET

# Variável Global de Parâmetros
CURRENT_PARAMS = config.DEFAULT_PARAMS
SLIPPAGE_HISTORY = [] 

def load_adaptive_params():
    """Carrega parâmetros baseados no Regime de Mercado E aplica Walk-Forward (PONTOS 1 & 7).
    Retorna: regime_str, IBOV_Price, IBOV_MA200
    """
    global CURRENT_PARAMS
    
    # get_market_regime retorna 4 valores: regime_str, curr, ma200, vix_br
    regime_data = utils.get_market_regime()
    regime_str = regime_data[0]
    px_ibov = regime_data[1]
    ma_ibov = regime_data[2]
    
    file_map = {
        "STRONG_BULL": config.PARAMS_STRONG_BULL,
        "BULL": config.PARAMS_BULL,
        "SIDEWAYS": config.PARAMS_SIDEWAYS,
        "BEAR": config.PARAMS_BEAR,
        "CRISIS": config.PARAMS_CRISIS,
    }
    
    base_file = file_map.get(regime_str, config.PARAMS_BULL)

    # 1. Lógica Walk-Forward (Removida por brevidade, mas deve existir)
    # ...
    
    # 2. Fallback 
    if os.path.exists(base_file):
        with open(base_file, 'r') as f:
            CURRENT_PARAMS = json.load(f)
            CURRENT_PARAMS['regime'] = regime_str
    else:
        CURRENT_PARAMS = config.DEFAULT_PARAMS.copy()
        CURRENT_PARAMS['regime'] = regime_str
        
    return regime_str, px_ibov, ma_ibov

# ==================== FUNÇÕES DE DISPLAY E STATUS ====================

def display_optimized_params():
    """Formata os parâmetros do regime atual para display."""
    params = CURRENT_PARAMS
    
    output = f"\n{AZUL}=== PARÂMETROS OTIMIZADOS ({params.get('regime', 'Default').upper()}) ==={RESET}\n"
    output += "-" * 110 + "\n"
    
    output += f"{'SIDE PREFERENCIAL':<20}: {params.get('side', 'NEUTRO'):<10}"
    output += f"{'SCORE OT. (R)':<18}: {params.get('score', 0.0):<10.2f}\n"
    
    output += f"{'EMA FAST/SLOW':<20}: {params.get('ema_fast', 0)}/{params.get('ema_slow', 0):<10}"
    output += f"{'RSI LEVEL (Zona)':<18}: {params.get('rsi_level', 0):<10}\n"
    
    output += f"{'ADX MÍNIMO':<20}: {params.get('adx_min', 0):<10}"
    output += f"{'MOMENTUM MÍNIMO':<18}: {params.get('momentum_min', 0.0):<10.2f}\n"
    
    output += "-" * 110
    return output

def analisar_carteira_detalhada():
    """Busca posições abertas e retorna string formatada (Função do usuário, ajustada)."""
    positions = mt5.positions_get()
    
    output = f"\n{AZUL}=== POSIÇÕES ABERTAS ({len(positions)}) ==={RESET}\n"
    output += "-" * 110 + "\n"
    
    if not positions:
        output += f"{AMARELO}Nenhuma posição aberta.{RESET}\n"
        output += "-" * 110
        return output
    
    header = f"{'ATIVO':<10} {'TIPO':<8} {'VOLUME':<8} {'PREÇO ENTRADA':<16} {'P&L R$':<12} {'P&L %':<8} {'SL':<12} {'TP':<12}"
    output += header + "\n"
    output += "-" * 110 + "\n"
    
    acc = mt5.account_info()
    balance = acc.balance if acc else 1.0 
    
    for pos in positions:
        sym = pos.symbol
        pos_type = "COMPRA" if pos.type == mt5.POSITION_TYPE_BUY else "VENDA"
        type_color = VERDE if pos.type == mt5.POSITION_TYPE_BUY else VERMELHO
        
        pl_rs = pos.profit
        pl_pct = (pos.profit / balance) * 100
        pl_color = VERDE if pl_rs >= 0 else VERMELHO
        
        # Uso das variáveis de cor importadas diretamente (VERDE, RESET, etc.)
        line = f"{sym:<10} {type_color + pos_type + RESET:<8} {pos.volume:<8.0f} "
        line += f"{pos.price_open:<16.4f} "
        line += f"{pl_color}R$ {pl_rs:,.2f}{RESET:<12} {pl_color}{pl_pct:+.2f}%{RESET:<8} "
        line += f"{pos.sl:<12.4f} {pos.tp:<12.4f}"
        output += line + "\n"
    
    output += "-" * 110
    return output

# ==================== EXECUÇÃO E SINALIZAÇÃO ====================

def execute_iceberg_order(symbol, type_order, total_volume, price_limit, sl, tp):
    """
    Executa ordem fatiada (Iceberg) com rastreamento de Slippage (PONTO 5).
    ... [Lógica omitida por brevidade, assumida como funcional] ...
    """
    global SLIPPAGE_HISTORY
    
    if not config.USE_ICEBERG_EXECUTION or total_volume <= 100:
        req = { "action": mt5.TRADE_ACTION_DEAL, "symbol": symbol, "volume": float(total_volume),
                "type": type_order, "price": price_limit, "type_filling": mt5.ORDER_FILLING_IOC,
                "sl": sl, "tp": tp, "magic": 20251230, "comment": f"Direct_Entry_{CURRENT_PARAMS['regime']}"
        }
        res = mt5.order_send(req)
        SLIPPAGE_HISTORY.append(0.0) 
        SLIPPAGE_HISTORY = SLIPPAGE_HISTORY[-10:] 
        return res
        
    
    logger.info(f"{AZUL}❄️ ICEBERG START: {symbol} Total: {total_volume} @ {price_limit:.2f}{RESET}")
    
    avg_slippage = sum(SLIPPAGE_HISTORY) / max(1, len(SLIPPAGE_HISTORY))
    # Cálculo chunk_size (assumido: (total_volume // config.ICEBERG_SPLIT // 100) * 100)
    chunk_size = 1000 # Valor fixo para mock, usar a lógica real do config

    if avg_slippage > 8.0: 
        reduction_factor = 0.70 
        chunk_size = (int(chunk_size * reduction_factor) // 100) * 100
        logger.warning(f"Alto Slippage ({avg_slippage:.2f} bps) -> Reduzindo Iceberg chunk size para {chunk_size}")

    if chunk_size < 100: chunk_size = 100
    
    remaining = total_volume
    total_value_executed = 0.0
    total_cash_executed = 0.0
    
    while remaining > 0:
        current_qty = min(chunk_size, remaining)
        
        req = {
            "action": mt5.TRADE_ACTION_DEAL, "symbol": symbol, "volume": float(current_qty),
            "type": type_order, "price": price_limit, "type_filling": mt5.ORDER_FILLING_IOC,
            "sl": sl, "tp": tp, "magic": 20251230, "comment": f"Iceberg_Leg_{CURRENT_PARAMS['regime']}"
        }
        res = mt5.order_send(req)
        
        if res.retcode == mt5.TRADE_RETCODE_DONE:
            vwap_executado = res.price 
            total_value_executed += vwap_executado * current_qty
            total_cash_executed += current_qty
            
        remaining -= current_qty
        
        sleep_time = random.uniform(0.5, 2.0) # Uso de valores mock se config.ICEBERG_MIN_DELAY não existir
        time.sleep(sleep_time)
        
    
    if total_cash_executed > 0:
        overall_vwap = total_value_executed / total_cash_executed
        slippage_cash = abs(overall_vwap - price_limit)
        slippage_bps = (slippage_cash / price_limit) * 10000 
        SLIPPAGE_HISTORY.append(slippage_bps)
        SLIPPAGE_HISTORY = SLIPPAGE_HISTORY[-10:] 
        logger.info(f"{AZUL}❄️ ICEBERG END: Executado {total_volume} @ VWAP {overall_vwap:.2f} | Slippage: {slippage_bps:.2f} bps{RESET}")
    else:
        SLIPPAGE_HISTORY.append(0.0)
        SLIPPAGE_HISTORY = SLIPPAGE_HISTORY[-10:]

    return res

def avaliar_ativo(symbol):
    """
    MOCK/PLACEHOLDER: Simula a avaliação de um ativo.
    **SUBSTITUA ESTA FUNÇÃO PELA SUA LÓGICA DE GERAÇÃO DE SINAL (Alpha).**
    """
    tick = mt5.symbol_info_tick(symbol)
    price = tick.ask if tick else 0.0
    
    # Mocks para popular a tabela
    is_buy = random.choice([True, False])
    score = random.uniform(80.0, 150.0)
    
    if score < 90.0: 
        action = "NEUTRO"
    else: 
        action = "COMPRA" if is_buy else "VENDA"
        
    rsi_val = random.uniform(30.0, 75.0)
    adx_val = random.uniform(15.0, 35.0)

    # Ajustes para os mocks parecerem mais reais
    if action == "COMPRA":
        rsi_val = random.uniform(55.0, 75.0)
    elif action == "VENDA":
        rsi_val = random.uniform(25.0, 45.0)
    
    return {
        "symbol": symbol,
        "action": action,
        "score": score,
        "data": {
            "price": price,
            "RSI": rsi_val,
            "EMA_FAST": price * random.uniform(0.995, 1.005),
            "EMA_SLOW": price * random.uniform(0.99, 1.01),
            "ADX": adx_val,
        },
        "motivo": f"{action} Score {score:.1f} > Min."
    }

# ==================== CICLO PRINCIPAL (Novo Painel de Controle) ====================

def ciclo_principal():
    """
    Função principal que faz o loop, o scan, a execução e o display no console.
    """
    # 1. LIMPA A TELA
    os.system('cls' if os.name == 'nt' else 'clear') 
    
    # 2. CABEÇALHO GLOBAL
    print(f"{AZUL}=== BOT ELITE 2026 PRO (INSTITUCIONAL KERNEL) ==={RESET}")
    
    # 3. REGIME DE MERCADO
    # load_adaptive_params retorna (regime_str, px_ibov, ma_ibov)
    regime_str, px_ibov, ma_ibov = load_adaptive_params() 
    regime_alta = px_ibov >= ma_ibov and regime_str not in ["SIDEWAYS", "BEAR", "CRISIS"]

    regime_txt = f"{VERDE}BULLISH (>MM200){RESET}" if regime_alta else f"{VERMELHO}BEARISH/LATERAL/CRISIS{RESET}"
    print(f"Mercado: {regime_txt} | IBOV: {px_ibov:,.0f} (MM200: {ma_ibov:,.0f}) | Regime: {ROXO}{regime_str}{RESET}")

    # 4. RESUMO FINANCEIRO
    account_info = mt5.account_info()
    if account_info is None:
        logger.error(f"{VERMELHO}Falha ao obter informações da conta MT5.{RESET}")
        return
    
    # Lógica do Equity Drop (para o Circuit Breaker)
    utils.EQUITY_DROP_HISTORY.append({'time': datetime.datetime.now(), 'equity': account_info.equity})
    if len(utils.EQUITY_DROP_HISTORY) > 2:
        utils.EQUITY_DROP_HISTORY.popleft()

    # Obtém o PNL
    pl_reais, pl_pct = utils.get_daily_profit_loss()

    print(f"\n{AZUL}=== RESUMO FINANCEIRO (Conta: {account_info.login}) ==={RESET}")
    print("-" * 110)
    
    pl_color = VERDE if pl_reais >= 0 else VERMELHO
    
    print(f"{'CAPITAL TOTAL (EQUITY)':<30}: R$ {account_info.equity:,.2f}")
    print(f"{'DINHEIRO DISPONÍVEL (LIVRE)':<30}: R$ {account_info.margin_free:,.2f}")
    print(f"{'INVESTIDO (MARGEM EM USO)':<30}: R$ {account_info.margin:,.2f}")
    
    print(f"{'PERDAS/LUCROS DIÁRIO (P&L)':<30}: {pl_color}R$ {pl_reais:,.2f} ({pl_pct:+.2f}%){RESET}")
    print("-" * 110)

    # 5. PARÂMETROS OTIMIZADOS
    print(display_optimized_params())
    
    # 6. RELATÓRIO DA CARTEIRA
    print(analisar_carteira_detalhada())

    # 7. SCAN DE OPORTUNIDADES
    
    try:
        # Usa uma lista mock se config.CANDIDATOS_BASE não existir
        candidatos_simulados = getattr(config, 'CANDIDATOS_BASE', ["PETR4", "VALE3", "ITUB4", "BBDC4", "ABEV3", "RENT3", "WEGE3"])
        with ThreadPoolExecutor(max_workers=10) as executor:
            resultados = list(executor.map(avaliar_ativo, candidatos_simulados))
    except Exception as e:
        logger.error(f"Erro no scan paralelo: {e}")
        resultados = []
    
    candidatos_validos = [r for r in resultados if r.get("status") != "ERRO" and r['data']['price'] > 0]
    candidatos_validos.sort(key=lambda x: x["score"], reverse=True)
    
    candidatos_para_display = candidatos_validos[:20]
    oportunidades_para_execucao = [r for r in candidatos_para_display if r["action"] in ["COMPRA", "VENDA"]]
    
    current_rsi_level = CURRENT_PARAMS.get('rsi_level', 70)
    current_adx_min = CURRENT_PARAMS.get('adx_min', 20)
    
    print(f"\n{AZUL}=== TOP {len(candidatos_para_display)} CANDIDATOS (Ordenados por Score) ==={RESET}")
    print("-" * 110)
    
    header = f"{'ATIVO':<10} {'AÇÃO':<8} {'SCORE':<8} {'RSI':<10} {'EMA FAST':<10} {'EMA SLOW':<10} {'ADX':<10} {'MOTIVO'}"
    print(header)
    print("-" * 110)
    
    for op in candidatos_para_display:
        sym = op["symbol"]
        acao = op["action"]
        score = op["score"]
        d = op["data"]
        motivo = op["motivo"]

        # Lógica de Cores para Indicadores
        ema_f_val = d.get('EMA_FAST', 0.0)
        ema_s_val = d.get('EMA_SLOW', 0.0)
        ema_f_color = AMARELO
        if (acao == "COMPRA" and ema_f_val > ema_s_val) or (acao == "VENDA" and ema_f_val < ema_s_val):
            ema_f_color = VERDE
        elif (acao == "COMPRA" and ema_f_val < ema_s_val) or (acao == "VENDA" and ema_f_val > ema_s_val):
            ema_f_color = VERMELHO
            
        rsi_color = AMARELO
        rsi_val = d.get('RSI', 50.0)
        
        if (acao == "COMPRA" and rsi_val > current_rsi_level) or (acao == "VENDA" and rsi_val < (100 - current_rsi_level)):
            rsi_color = VERMELHO
        elif (acao == "COMPRA" and rsi_val > 50 and rsi_val <= current_rsi_level) or (acao == "VENDA" and rsi_val < 50 and rsi_val >= (100 - current_rsi_level)):
            rsi_color = VERDE
            
        adx_color = VERMELHO
        adx_val = d.get('ADX', 0.0)
        if adx_val >= current_adx_min:
            adx_color = VERDE

        score_color = VERDE if acao != "NEUTRO" else AMARELO
        acao_color = VERDE if acao == "COMPRA" else (VERMELHO if acao == "VENDA" else RESET)
        
        rsi_formatted = f"{rsi_color}{rsi_val:.1f}{RESET}"
        ema_f_formatted = f"{ema_f_color}{ema_f_val:.2f}{RESET}"
        ema_s_formatted = f"{ema_s_val:.2f}"
        adx_formatted = f"{adx_color}{adx_val:.1f}{RESET}"
        score_formatted = f"{score_color}{score:.1f}{RESET}"

        line = f"{sym:<10} {acao_color + acao + RESET:<8} {score_formatted:<8} {rsi_formatted:<10} {ema_f_formatted:<10} {ema_s_formatted:<10} {adx_formatted:<10} {motivo}"
        print(line)
    
    print("-" * 110)
    
    # 8. Execução de Ordens (Top 1)
    if oportunidades_para_execucao:
        target = oportunidades_para_execucao[0]
        sym = target['symbol']
        
        if not mt5.positions_get(symbol=sym):
            
            is_buy = (target['action'] == "COMPRA")
            price = target['data']['price']
            
            # Chama a função de ATR corrigida em utils.py
            atr = utils.calculate_asset_atr(sym) 
            
            if atr > 0:
                sl_dist = atr * CURRENT_PARAMS.get("sl_atr_mult", 2.0)
                tp_dist = sl_dist * CURRENT_PARAMS.get("tp_mult", 1.5)
                
                sl = price - sl_dist if is_buy else price + sl_dist
                tp = price + tp_dist if is_buy else price - tp_dist
                
                sl = max(0.01, sl) if sl_dist > 0 else 0.0
                
                vol = utils.calcular_tamanho_posicao(sym, sl, is_buy)
                
                if vol > 0:
                    type_ord = mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL
                    
                    logger.info(f"ENTRADA: {sym} | {target['action']} | Vol: {vol} | SL: {sl:.2f} | TP: {tp:.2f}")
                    execute_iceberg_order(sym, type_ord, vol, price, sl, tp)
                    utils.register_trade(sym, target['action'], price, vol, "ENTRY", sl, tp)
                else:
                    logger.info(f"ENTRADA IGNORADA: {sym} | Vol calculado 0 (Risco muito alto/Vol Targeting).")
            else:
                logger.warning(f"ENTRADA IGNORADA: {sym} | ATR falhou (0) para cálculo do SL.")


def bot_loop_wrapper():
    logger.info(f"{VERDE}Bot Iniciado - Modo Institucional (Kernel V3){RESET}")
    
    # 1. Inicia Trailing em Thread Separada
    t_trail = threading.Thread(target=utils.trailing_stop_service, daemon=True)
    t_trail.start()
    
    while True:
        try:
            if os.path.exists("STOP.CMD"): 
                logger.warning("Arquivo STOP.CMD encontrado. Encerrando bot.")
                break
                
            acc_info = mt5.account_info()
            if acc_info is None:
                logger.error("Falha ao obter account_info. Tentando reconexão.")
                utils.check_mt5_connection()
                time.sleep(config.CHECK_INTERVAL_SLOW)
                continue
            
            # 2. Segurança & Robustez (Heartbeat/Panic)
            tick_data = mt5.symbol_info_tick(config.IBOV_SYMBOL)
            # Correção: O Circuit Breaker precisa do acc_info para verificar o PNL
            if utils.check_circuit_breakers(acc_info, tick_data):
                break 

            # 3. Verifica Horário de Operação (Usa datetime.time() para comparação)
            current_time = datetime.datetime.now().time()
            if not (config.START_TIME <= current_time <= config.END_TIME):
                # Se estiver fora do horário, apenas imprime um resumo rápido sem a lógica completa
                print(f"FORA DE HORÁRIO: {current_time.strftime('%H:%M:%S')} | Equity: R$ {acc_info.equity:,.2f}", end='\r')
                time.sleep(60)
                continue
                
            # 4. Executa o Ciclo Principal (Display e Execução)
            ciclo_principal()
            
        except Exception as e:
            # Erro no loop principal agora mostrará no log e no console (via logger)
            logger.error(f"{VERMELHO}Erro no loop principal: {e}{RESET}")

        time.sleep(config.CHECK_INTERVAL_SLOW)

if __name__ == '__main__':
    if utils.check_mt5_connection():
        # Assumindo que config.TIMEFRAME_MT5, config.IBOV_SYMBOL e outras variáveis de config.py existem.
        bot_loop_wrapper()