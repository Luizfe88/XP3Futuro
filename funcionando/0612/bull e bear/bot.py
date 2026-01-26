# bot.py – EXECUTOR INSTITUCIONAL B3
import MetaTrader5 as mt5
import time
import os
import json
import threading
import random
import datetime
from concurrent.futures import ThreadPoolExecutor
import pandas as pd # Necessário para a execução da ordem Iceberg, caso utils seja chamado antes de bot.

import config
import utils
from utils import logger, Fore, Style

# Variável Global de Parâmetros
CURRENT_PARAMS = config.DEFAULT_PARAMS

def load_adaptive_params():
    """Carrega parâmetros baseados no Regime de Mercado (Bull/Bear/Sideways)."""
    global CURRENT_PARAMS
    # O get_market_regime agora retorna dados do IBOV, mas o regime string é o primeiro item
    regime = utils.get_market_regime()[0] 
    
    file_map = {
        "BULL": config.PARAMS_BULL,
        "BEAR": config.PARAMS_BEAR,
        "SIDEWAYS": config.PARAMS_SIDE
    }
    
    target_file = file_map.get(regime, config.PARAMS_BULL)
    
    if os.path.exists(target_file):
        with open(target_file, 'r') as f:
            CURRENT_PARAMS = json.load(f)
            # Adiciona o regime de volta para o display
            CURRENT_PARAMS['regime'] = regime
            logger.info(f"Regime: {regime} -> Params carregados: {target_file}")
    else:
        logger.warning(f"Regime {regime}: Arquivo {target_file} não encontrado. Usando Default.")
        CURRENT_PARAMS = config.DEFAULT_PARAMS.copy()
        CURRENT_PARAMS['regime'] = "DEFAULT"
        
    return regime

# ==================== FUNÇÕES DE DISPLAY ====================

def display_optimized_params():
    """Formata e retorna a string de parâmetros otimizados ativos."""
    global CURRENT_PARAMS
    params = CURRENT_PARAMS
    
    side = params.get('side', 'COMPRA/VENDA')
    
    output = f"\n{utils.AZUL}=== PARÂMETROS OTIMIZADOS ATIVOS (Regime: {params.get('regime', 'DEFAULT').upper()}) ==={utils.RESET}\n"
    output += "-" * 110 + "\n"
    output += f"{'SIDE OTIMIZADO':<20}: {utils.VERDE + side + utils.RESET}\n"
    output += f"{'EMA FAST':<20}: {params.get('ema_fast', config.DEFAULT_PARAMS['ema_fast'])}\n"
    output += f"{'EMA SLOW':<20}: {params.get('ema_slow', config.DEFAULT_PARAMS['ema_slow'])}\n"
    output += f"{'RSI NÍVEL':<20}: {params.get('rsi_level', 70)} (Penalidade)\n"
    output += f"{'ADX MÍNIMO':<20}: {params.get('adx_min', 20)}\n"
    output += f"{'TP MULTIPLIER':<20}: {params.get('tp_mult', 1.5):.1f} R\n"
    output += f"{'SL ATR MULT':<20}: {params.get('sl_atr_mult', 2.0):.1f} ATR\n"
    output += "-" * 110
    return output

def analisar_carteira_detalhada():
    """Busca posições abertas e retorna string formatada."""
    positions = mt5.positions_get()
    
    output = f"\n{utils.AZUL}=== POSIÇÕES ABERTAS ({len(positions)}) ==={utils.RESET}\n"
    output += "-" * 110 + "\n"
    
    if not positions:
        output += f"{utils.AMARELO}Nenhuma posição aberta.{utils.RESET}\n"
        output += "-" * 110
        return output
    
    header = f"{'ATIVO':<10} {'TIPO':<8} {'VOLUME':<8} {'PREÇO ENTRADA':<16} {'P&L R$':<12} {'P&L %':<8} {'SL':<12} {'TP':<12}"
    output += header + "\n"
    output += "-" * 110 + "\n"
    
    acc = mt5.account_info()
    balance = acc.balance if acc else 1.0 # Fallback
    
    for pos in positions:
        sym = pos.symbol
        pos_type = "COMPRA" if pos.type == mt5.POSITION_TYPE_BUY else "VENDA"
        type_color = utils.VERDE if pos.type == mt5.POSITION_TYPE_BUY else utils.VERMELHO
        
        pl_rs = pos.profit
        pl_pct = (pos.profit / balance) * 100
        pl_color = utils.VERDE if pl_rs >= 0 else utils.VERMELHO
        
        line = f"{sym:<10} {type_color + pos_type + utils.RESET:<8} {pos.volume:<8.0f} "
        line += f"{pos.price_open:<16.4f} "
        line += f"{pl_color}R$ {pl_rs:,.2f}{utils.RESET:<12} {pl_color}{pl_pct:+.2f}%{utils.RESET:<8} "
        line += f"{pos.sl:<12.4f} {pos.tp:<12.4f}"
        output += line + "\n"
    
    output += "-" * 110
    return output

# ==================== EXECUÇÃO INSTITUCIONAL (ICEBERG) ====================

def execute_iceberg_order(symbol, type_order, total_volume, price_limit, sl, tp):
    """
    Executa ordem fatiada (TWAP/Iceberg) para minimizar impacto e slippage.
    (Função mantida, apenas renomeada de 'execute_iceberg_order' para 'execute_iceberg_order' - sem alteração de lógica)
    """
    if not config.USE_ICEBERG_EXECUTION or total_volume <= 100:
        # Execução Direta se pequeno
        req = {
            "action": mt5.TRADE_ACTION_DEAL, "symbol": symbol, "volume": float(total_volume),
            "type": type_order, "price": price_limit, "sl": sl, "tp": tp,
            "magic": 20251230, "type_filling": mt5.ORDER_FILLING_IOC
        }
        return mt5.order_send(req)

    logger.info(f"❄️ ICEBERG START: {symbol} Total: {total_volume}")
    chunk_size = (total_volume // config.ICEBERG_SPLIT // 100) * 100
    if chunk_size < 100: chunk_size = 100
    
    remaining = total_volume
    
    while remaining > 0:
        current_qty = min(chunk_size, remaining)
        
        # Limit Order inside Spread (Melhora execução)
        tick = mt5.symbol_info_tick(symbol)
        my_price = tick.ask - 0.01 if type_order == mt5.ORDER_TYPE_BUY else tick.bid + 0.01 
        
        req = {
            "action": mt5.TRADE_ACTION_PENDING, # Limit
            "symbol": symbol, "volume": float(current_qty),
            "type": mt5.ORDER_TYPE_BUY_LIMIT if type_order == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_SELL_LIMIT,
            "price": my_price, "sl": sl, "tp": tp,
            "magic": 20251230, "comment": "Iceberg_Leg"
        }
        
        res = mt5.order_send(req)
        
        if res.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"Iceberg Leg enviada: {current_qty} @ {my_price}")
            
            # Espera preencher ou timeout
            start_wait = time.time()
            filled = False
            while time.time() - start_wait < config.ORDER_TIMEOUT_SEC:
                orders = mt5.orders_get(ticket=res.order)
                if not orders: # Sumiu = Executou ou cancelou
                    filled = True
                    break
                time.sleep(1)
            
            if not filled:
                # Cancela e manda a mercado
                mt5.order_send({"action": mt5.TRADE_ACTION_REMOVE, "order": res.order})
                logger.warning("Timeout Iceberg -> Convertendo para Market")
                req["action"] = mt5.TRADE_ACTION_DEAL
                req["type"] = type_order 
                req["type_filling"] = mt5.ORDER_FILLING_IOC
                mt5.order_send(req)
                
            remaining -= current_qty
            
            # Random Delay para mascarar algoritmo
            sleep_time = random.uniform(config.ICEBERG_MIN_DELAY, config.ICEBERG_MAX_DELAY)
            time.sleep(sleep_time)
            
        else:
            logger.error(f"Erro Iceberg: {res.comment}")
            break

# ==================== LÓGICA DE DECISÃO (RENOMEADO) ====================

def avaliar_ativo(symbol):
    """Calcula Score e retorna decisão e dados detalhados, incluindo o motivo."""
    score, data, action = utils.get_asset_score(symbol, CURRENT_PARAMS)
    
    res = {
        "symbol": symbol, 
        "score": score, 
        "action": action, # COMPRA, VENDA ou NEUTRO
        "data": data, 
        "status": "VALIDO",
        "motivo": "Score abaixo do mínimo"
    }

    # Check for errors/no data (utils.get_asset_score returns 0, {}, "No Data" on error)
    if action == "No Data":
        res["status"] = "ERRO"
        res["motivo"] = "Dados MT5 insuficientes (Velho)"
        res["action"] = "NEUTRO"
        return res
        
    motivos = []
    
    # Lógica de Motivo (Ação)
    is_buy = (action == "COMPRA")
    
    # 1. Checa Tendência EMA
    ema_f = data['ema_fast']
    ema_s = data['ema_slow']
    
    if (is_buy and ema_f > ema_s) or (not is_buy and ema_f < ema_s):
        motivos.append("EMA Direcional")
    else:
        motivos.append("EMA Cruzando/Contrário")

    # 2. Checa Força ADX
    adx_min = CURRENT_PARAMS.get('adx_min', 20)
    if data['adx'] >= adx_min:
        motivos.append(f"ADX Forte ({data['adx']:.1f})")
    else:
        motivos.append(f"ADX Fraco ({data['adx']:.1f})")
        
    # 3. Checa Penalidade RSI
    rsi_penalty_level = CURRENT_PARAMS.get("rsi_level", 70) 
    rsi_val = data['rsi']
    
    if is_buy:
        if rsi_val > rsi_penalty_level:
            motivos.append(f"RSI SOBRECOMPRADO ({rsi_val:.1f})")
        elif rsi_val < (100 - rsi_penalty_level):
            motivos.append(f"RSI SOBREVENDIDO ({rsi_val:.1f}) - Ok")
        else:
            motivos.append("RSI Neutro")
    else: # VENDA
        if rsi_val < (100 - rsi_penalty_level):
            motivos.append(f"RSI SOBREVENDIDO ({rsi_val:.1f})")
        elif rsi_val > rsi_penalty_level:
            motivos.append(f"RSI SOBRECOMPRADO ({rsi_val:.1f}) - Ok")
        else:
            motivos.append("RSI Neutro")
            
    res["motivo"] = " | ".join(motivos)
    
    # Filtro final de entrada (se o score final não for atingido, a ação é NEUTRO)
    if score < config.MIN_SCORE_ENTRY:
        res["action"] = "NEUTRO"
        res["motivo"] = f"Score {score:.1f} < Mínimo {config.MIN_SCORE_ENTRY:.1f}"
    
    # Adicionar dados formatados para o display
    res['data']['RSI'] = res['data']['rsi']
    res['data']['ADX'] = res['data']['adx']
    res['data']['EMA_FAST'] = res['data']['ema_fast']
    res['data']['EMA_SLOW'] = res['data']['ema_slow']
    
    return res

# ==================== NOVO CICLO PRINCIPAL COM DISPLAY ====================

def ciclo_principal():
    """
    Função principal que faz o loop, o scan, a execução e o display no console.
    """
    # 1. LIMPA A TELA
    os.system('cls' if os.name == 'nt' else 'clear') 
    
    # 2. CABEÇALHO GLOBAL
    print(f"{utils.AZUL}=== BOT ELITE 2026 PRO (INSTITUCIONAL KERNEL) ==={utils.RESET}")
        
    # 3. REGIME DE MERCADO
    regime_str, px_ibov, ma_ibov = utils.get_market_regime()
    regime_alta = px_ibov >= ma_ibov and regime_str != "SIDEWAYS"

    regime_txt = f"{utils.VERDE}BULLISH (>MM200){utils.RESET}" if regime_alta else f"{utils.VERMELHO}BEARISH/LATERAL{utils.RESET}"
    print(f"Mercado: {regime_txt} | IBOV: {px_ibov:,.0f} (MM200: {ma_ibov:,.0f})")

    # 4. RESUMO FINANCEIRO
    account_info = mt5.account_info()
    if account_info is None:
        logger.error(f"{utils.VERMELHO}Falha ao obter informações da conta MT5.{utils.RESET}")
        return
    
    # É preciso rodar o load_adaptive_params para garantir que o CURRENT_PARAMS esteja fresco
    load_adaptive_params()

    pl_reais, pl_pct = utils.get_daily_profit_loss()

    print(f"\n{utils.AZUL}=== RESUMO FINANCEIRO (Conta: {account_info.login}) ==={utils.RESET}")
    print("-" * 110)
    
    pl_color = utils.VERDE if pl_reais >= 0 else utils.VERMELHO
    
    print(f"{'CAPITAL TOTAL (EQUITY)':<30}: R$ {account_info.equity:,.2f}")
    print(f"{'DINHEIRO DISPONÍVEL (LIVRE)':<30}: R$ {account_info.margin_free:,.2f}")
    print(f"{'INVESTIDO (MARGEM EM USO)':<30}: R$ {account_info.margin:,.2f}")
    
    print(f"{'PERDAS/LUCROS DIÁRIO (P&L)':<30}: {pl_color}R$ {pl_reais:,.2f} ({pl_pct:+.2f}%){utils.RESET}")
    print("-" * 110)

    # 5. PARÂMETROS OTIMIZADOS
    print(display_optimized_params())
    
    # 6. RELATÓRIO DA CARTEIRA
    print(analisar_carteira_detalhada())

    # 7. SCAN DE OPORTUNIDADES
    
    # Scan Paralelo
    with ThreadPoolExecutor(max_workers=10) as executor:
        resultados = list(executor.map(avaliar_ativo, config.CANDIDATOS_BASE))
    
    # Filtra erros, ordena por score e limita o display
    candidatos_validos = [r for r in resultados if r["status"] != "ERRO"]
    candidatos_validos.sort(key=lambda x: x["score"], reverse=True)
    
    # Limita o display aos top 20
    candidatos_para_display = candidatos_validos[:20]
    
    # Lista para Execução: Apenas os que estão no TOP 20 e têm sinal de entrada
    oportunidades_para_execucao = [r for r in candidatos_para_display if r["action"] in ["COMPRA", "VENDA"]]
    
    # Parâmetros otimizados (para colorir)
    current_rsi_level = CURRENT_PARAMS.get('rsi_level', 70)
    current_adx_min = CURRENT_PARAMS.get('adx_min', 20)
    
    # Display Resumido de Oportunidades
    print(f"\n{utils.AZUL}=== TOP {len(candidatos_para_display)} CANDIDATOS (Ordenados por Score) ==={utils.RESET}")
    print("-" * 110)
    
    # NOVO CABEÇALHO DA TABELA DE OPORTUNIDADES
    header = f"{'ATIVO':<10} {'AÇÃO':<8} {'SCORE':<8} {'RSI':<10} {'EMA FAST':<10} {'EMA SLOW':<10} {'ADX':<10} {'MOTIVO'}"
    print(header)
    print("-" * 110)
    
    for op in candidatos_para_display:
        sym = op["symbol"]
        acao = op["action"]
        score = op["score"]
        d = op["data"]
        motivo = op["motivo"]

        # --- Lógica de Cores para Indicadores (Adaptada) ---
        
        # 1. EMA (Direção da Tendência)
        ema_f_val = d.get('EMA_FAST', 0.0)
        ema_s_val = d.get('EMA_SLOW', 0.0)
        ema_f_color = utils.AMARELO
        if (acao == "COMPRA" and ema_f_val > ema_s_val) or \
           (acao == "VENDA" and ema_f_val < ema_s_val):
            ema_f_color = utils.VERDE
        elif (acao == "COMPRA" and ema_f_val < ema_s_val) or \
             (acao == "VENDA" and ema_f_val > ema_s_val):
            ema_f_color = utils.VERMELHO
            
        # 2. RSI (Condição de Sobrevenda/Sobrecompra)
        rsi_color = utils.AMARELO
        rsi_val = d.get('RSI', 50.0)
        
        # Vermelho: Extremo (zona de penalidade)
        if (acao == "COMPRA" and rsi_val > current_rsi_level) or \
           (acao == "VENDA" and rsi_val < (100 - current_rsi_level)):
            rsi_color = utils.VERMELHO
        # Verde: Zona Saudável de Momentum
        elif (acao == "COMPRA" and rsi_val > 50 and rsi_val <= current_rsi_level) or \
             (acao == "VENDA" and rsi_val < 50 and rsi_val >= (100 - current_rsi_level)):
            rsi_color = utils.VERDE
            
        # 3. ADX (Força da Tendência)
        adx_color = utils.VERMELHO
        adx_val = d.get('ADX', 0.0)
        if adx_val >= current_adx_min:
            adx_color = utils.VERDE

        # --- Formatação dos Dados ---
        score_color = utils.VERDE if acao != "NEUTRO" else utils.AMARELO
        acao_color = utils.VERDE if acao == "COMPRA" else (utils.VERMELHO if acao == "VENDA" else utils.RESET)
        
        rsi_formatted = f"{rsi_color}{rsi_val:.1f}{utils.RESET}"
        ema_f_formatted = f"{ema_f_color}{ema_f_val:.2f}{utils.RESET}"
        ema_s_formatted = f"{ema_s_val:.2f}"
        adx_formatted = f"{adx_color}{adx_val:.1f}{utils.RESET}"
        score_formatted = f"{score_color}{score:.1f}{utils.RESET}"

        # Linha formatada com as cores
        line = f"{sym:<10} {acao_color + acao + utils.RESET:<8} {score_formatted:<8} {rsi_formatted:<10} {ema_f_formatted:<10} {ema_s_formatted:<10} {adx_formatted:<10} {motivo}"
        print(line)
    
    print("-" * 110)
    
    # 8. Execução de Ordens (Top 1)
    if oportunidades_para_execucao:
        target = oportunidades_para_execucao[0]
        sym = target['symbol']
        
        # Checa se já tem posição
        if not mt5.positions_get(symbol=sym):
            
            is_buy = (target['action'] == "COMPRA")
            price = target['data']['price']
            
            atr = utils.calculate_asset_atr(sym) 
            
            sl_dist = atr * CURRENT_PARAMS.get("sl_atr_mult", 2.0)
            tp_dist = sl_dist * CURRENT_PARAMS.get("tp_mult", 1.5)
            
            sl = price - sl_dist if is_buy else price + sl_dist
            tp = price + tp_dist if is_buy else price - tp_dist
            
            # Validação: SL deve ser positivo.
            sl = max(0.01, sl) if sl_dist > 0 else 0.0
            
            vol = utils.calcular_tamanho_posicao(sym, sl, is_buy)
            
            if vol > 0:
                type_ord = mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL
                
                logger.info(f"ENTRADA: {sym} | {target['action']} | Vol: {vol} | SL: {sl:.2f} | TP: {tp:.2f}")
                execute_iceberg_order(sym, type_ord, vol, price, sl, tp)

# ==================== FUNÇÃO WRAPPER PRINCIPAL (LOOP) ====================

def bot_loop_wrapper():
    logger.info("Bot Iniciado - Modo Institucional")
    
    # Inicia Trailing em Thread Separada
    t_trail = threading.Thread(target=utils.trailing_stop_service, daemon=True)
    t_trail.start()
    
    while True:
        if os.path.exists("STOP.CMD"): 
            logger.warning("Arquivo STOP.CMD encontrado. Encerrando bot.")
            break
        
        # 1. Proteção (DD e Equity Peak)
        if utils.check_circuit_breakers(): continue
        
        # 2. Verifica Horário de Operação
        if not (config.START_TIME <= datetime.datetime.now().time() <= config.END_TIME):
            print(f"Fora de Horário: {datetime.datetime.now().strftime('%H:%M:%S')}", end='\r')
            time.sleep(60)
            continue
            
        # 3. Executa o Ciclo Principal (Display e Execução)
        try:
            ciclo_principal()
        except Exception as e:
            logger.error(f"Erro no ciclo principal: {e}")

        time.sleep(config.CHECK_INTERVAL_SLOW)

if __name__ == "__main__":
    if mt5.initialize():
        try:
            # Chama o wrapper do loop
            bot_loop_wrapper()
        except KeyboardInterrupt:
            logger.info("Bot encerrado via KeyboardInterrupt.")
            pass
        mt5.shutdown()