# bot.py – EXECUTOR INSTITUCIONAL (HEADLESS READY)
import MetaTrader5 as mt5
import time
import os
import sys
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import config
import utils
from utils import logger, VERDE, VERMELHO, AMARELO, AZUL, ROXO, BRANCO, RESET

# ==================== LÓGICA DE TRADING ====================

def avaliar_ativo(symbol):
    """Analisa um ativo e retorna veredito."""
    data, error = utils.get_asset_technical_data(symbol)
    
    # Garante que as chaves de controle (acao, status) estejam presentes no erro
    if error: return {"symbol": symbol, "status": "ERRO", "score": -1, "acao": "ERRO"} 
    
    pc = config.PARAMETROS_OTIMIZADOS_COMPRA
    pv = config.PARAMETROS_OTIMIZADOS_VENDA
    
    rejeicoes = []
    score_final = 0
    acao = "NEUTRO"
    
    # 1. Filtros Globais (Regime e Setor)
    regime_alta, _, _ = utils.get_market_regime()
    if config.USE_MARKET_REGIME_FILTER:
        if regime_alta and not utils.is_trading_time(): pass 
    
    # 2. Check Compra
    score_c = 0
    if data["EMA_FAST"] > data["EMA_SLOW"]:
        if data["RSI"] < pc["rsi_max"]:
            if data["momentum"] > pc["momentum_min"]:
                if data["ADX"] > config.ADX_MIN_THRESHOLD:
                    score_c = 100
                else: rejeicoes.append("ADX Fraco")
            else: rejeicoes.append("Momentum Baixo")
        else: rejeicoes.append("RSI Alto")
    else: rejeicoes.append("Tendência Baixa")
    
    # 3. Check Venda
    score_v = 0
    if data["EMA_FAST"] < data["EMA_SLOW"]:
        if data["RSI"] > pv["rsi_min"]:
            if data["momentum"] < pv["momentum_max_neg"]:
                if data["ADX"] > config.ADX_MIN_THRESHOLD:
                    score_v = 100
    
    # Decisão
    if score_c > 80 and regime_alta:
        acao = "COMPRA"
        score_final = score_c
    elif score_v > 80 and not regime_alta:
        acao = "VENDA"
        score_final = score_v
    else:
        acao = "AGUARDAR"
        
    return {
        "symbol": symbol,
        "status": "OK", 
        "acao": acao,
        "score": score_final,
        "data": data,
        "motivo": ", ".join(rejeicoes) if acao == "AGUARDAR" else "Setup Confirmado"
    }

def executar_ordem(sinal):
    sym = sinal["symbol"]
    lado = sinal["acao"]
    
    # Validações Finais
    if lado == "COMPRA":
        vol = utils.calcular_volume_inteligente(sym)
        if vol == 0: return # Sem saldo ou setor cheio
        sl, tp, _ = utils.calculate_atr_sl_tp(sym, sinal["data"]["price"], True)
        
        req = {
            "action": mt5.TRADE_ACTION_DEAL, "symbol": sym, "volume": float(vol),
            "type": mt5.ORDER_TYPE_BUY, "price": sinal["data"]["tick"].ask,
            "sl": sl, "tp": tp, "magic": 20251230, "comment": "ElitePro_V1",
            "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
    elif lado == "VENDA":
        vol = utils.calcular_volume_inteligente(sym)
        if vol == 0: return
        sl, tp, _ = utils.calculate_atr_sl_tp(sym, sinal["data"]["price"], False)
        
        req = {
            "action": mt5.TRADE_ACTION_DEAL, "symbol": sym, "volume": float(vol),
            "type": mt5.ORDER_TYPE_SELL, "price": sinal["data"]["tick"].bid,
            "sl": sl, "tp": tp, "magic": 20251230, "comment": "ElitePro_V1",
            "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_IOC,
        }
    
    res = mt5.order_send(req)
    if res.retcode == mt5.TRADE_RETCODE_DONE:
        logger.info(f"{VERDE}ORDEM EXECUTADA: {lado} {sym} Vol:{vol} @ {req['price']}{RESET}")
        utils.register_trade(sym, lado, req['price'], vol, "ENTRY", sl, tp)
        # Tenta tocar som se possível, senão ignora
        try:
            import winsound
            winsound.Beep(1000, 200)
        except: pass
    else:
        logger.error(f"Erro Ordem {sym}: {res.comment}")

# ==================== LÓGICA DE RELATÓRIOS (INTEGRADA) ====================

def analisar_carteira_detalhada():
    """Analisa as posições abertas e retorna a string formatada para impressão."""
    positions = mt5.positions_get()
    
    if not positions:
        return f"\n{AZUL}Nenhuma posição aberta na carteira.{RESET}"

    output = f"\n{ROXO}=== RELATÓRIO DETALHADO DA CARTEIRA ({len(positions)} POSIÇÕES) ==={RESET}\n"
    output += "-" * 110 + "\n"
    
    # Cabeçalho da tabela
    header = f"{'ATIVO':<10} {'TIPO':<6} {'PREÇO ENTRADA':<15} {'VOLUME':<8} {'RSI':<5} {'EMA RÁPIDA':<12} {'EMA LENTA':<10} {'ADX':<5} {'STATUS/MOTIVO'}"
    output += header + "\n"
    output += "-" * 110 + "\n"

    for pos in positions:
        symbol = pos.symbol
        side = "COMPRA" if pos.type == mt5.ORDER_TYPE_BUY else "VENDA"
        
        analise = avaliar_ativo(symbol)
        data = analise.get("data", {})
        acao_sugerida = analise.get("acao", "ERRO")
        motivo = analise.get("motivo", "Falha na análise")
        
        if (side == "COMPRA" and acao_sugerida == "COMPRA") or \
           (side == "VENDA" and acao_sugerida == "VENDA"):
            cor = VERDE
            status_motivo = "OK / Confirma Setup"
        else:
            cor = VERMELHO
            status_motivo = f"INVERSÃO / Bot sugere: {acao_sugerida} ({motivo})"

        rsi_val = f"{data.get('RSI', 0):.1f}"
        ema_f_val = f"{data.get('EMA_FAST', 0):.2f}"
        ema_s_val = f"{data.get('EMA_SLOW', 0):.2f}"
        adx_val = f"{data.get('ADX', 0):.1f}"
        
        line = f"{cor}{symbol:<10} {side:<6} {pos.price_open:<15.2f} {pos.volume:<8.0f} {rsi_val:<5} {ema_f_val:<12} {ema_s_val:<10} {adx_val:<5} {status_motivo}{RESET}"
        output += line + "\n"

    output += "-" * 110 + "\n"
    output += f"{VERDE}Status 'OK': A posição AINDA cumpre os critérios técnicos de entrada do robô.{RESET}\n"
    output += f"{VERMELHO}Status 'INVERSÃO': A posição NÃO cumpre mais os critérios (sinal contrário ou neutro).{RESET}\n"
    output += "-" * 110 + "\n"
    
    return output

def display_optimized_params():
    """Formata e retorna uma string com os parâmetros otimizados atuais."""
    output = f"\n{AZUL}=== PARÂMETROS OTIMIZADOS ATUAIS (by optimizer.py) ==={RESET}\n"
    output += "-" * 110 + "\n"
    
    # Parâmetros de Compra
    pc = config.PARAMETROS_OTIMIZADOS_COMPRA
    score_c = pc.get('score', 0.0)
    cor_c = VERDE if score_c > 0 else AMARELO
    
    output += f"{ROXO}COMPRA (Score: {cor_c}{score_c:.1f}{RESET}{ROXO}):{RESET}\n"
    output += f"  - EMA Rápida: {pc.get('ema_fast', '-')}, EMA Lenta: {pc.get('ema_slow', '-')}\n"
    output += f"  - RSI Máx: {pc.get('rsi_max', '-')}, Momentum Mín: {pc.get('momentum_min', '-')}\n"
    
    output += "-" * 110 + "\n"

    # Parâmetros de Venda
    pv = config.PARAMETROS_OTIMIZADOS_VENDA
    score_v = pv.get('score', 0.0)
    cor_v = VERDE if score_v > 0 else AMARELO
    
    output += f"{ROXO}VENDA (Score: {cor_v}{score_v:.1f}{RESET}{ROXO}):{RESET}\n"
    output += f"  - EMA Rápida: {pv.get('ema_fast', '-')}, EMA Lenta: {pv.get('ema_slow', '-')}\n"
    output += f"  - RSI Mín: {pv.get('rsi_min', '-')}, Momentum Máx Neg: {pv.get('momentum_max_neg', '-')}\n"
    
    output += "-" * 110 + "\n"
    return output


# ==================== CICLO E EXECUÇÃO PRINCIPAL (Integrado) ====================

def ciclo_principal():
    # 1. LIMPA A TELA
    os.system('cls' if os.name == 'nt' else 'clear') 
    
    # 2. CABEÇALHO GLOBAL
    print(f"{AZUL}=== BOT ELITE 2026 PRO (HEADLESS) ==={RESET}")
    print("Para parar, crie o arquivo 'STOP.CMD' na pasta.")
    
    # 3. REGIME DE MERCADO
    regime_alta, px_ibov, ma_ibov = utils.get_market_regime()
    regime_txt = f"{VERDE}BULLISH (>MM200){RESET}" if regime_alta else f"{VERMELHO}BEARISH (<MM200){RESET}"
    print(f"Mercado: {regime_txt} | IBOV: {px_ibov:.0f} (MM200: {ma_ibov:.0f})")

    # 4. RESUMO FINANCEIRO
    account_info = mt5.account_info()
    
    if account_info is None:
        logger.error(f"{VERMELHO}Falha ao obter informações da conta MT5.{RESET}")
        return
        
    pl_reais, pl_pct = utils.get_daily_profit_loss()

    print(f"\n{AZUL}=== RESUMO FINANCEIRO (Conta: {account_info.login}) ==={RESET}")
    print("-" * 110)
    
    pl_color = VERDE if pl_reais >= 0 else VERMELHO
    
    print(f"{'CAPITAL TOTAL (EQUITY)':<30}: R$ {account_info.equity:,.2f}")
    print(f"{'DINHEIRO DISPONÍVEL (LIVRE)':<30}: R$ {account_info.margin_free:,.2f}")
    print(f"{'INVESTIDO (MARGEM EM USO)':<30}: R$ {account_info.margin:,.2f}")
    
    print(f"{'PERDAS/LUCROS DIÁRIO (P&L)':<30}: {pl_color}R$ {pl_reais:,.2f} ({pl_pct:+.2f}%){RESET}")
    print("-" * 110)

    # 5. PARÂMETROS OTIMIZADOS (NOVO BLOCO)
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
    oportunidades_para_execucao = [r for r in candidatos_para_display if r["acao"] in ["COMPRA", "VENDA"]]
    
    # Parâmetros otimizados para colorir
    pc = config.PARAMETROS_OTIMIZADOS_COMPRA
    pv = config.PARAMETROS_OTIMIZADOS_VENDA
    
    # Display Resumido de Oportunidades
    print(f"\n{AZUL}=== TOP {len(candidatos_para_display)} CANDIDATOS (Ordenados por Score) ==={RESET}")
    print("-" * 110)
    
    # NOVO CABEÇALHO DA TABELA DE OPORTUNIDADES
    header = f"{'ATIVO':<10} {'AÇÃO':<8} {'SCORE':<8} {'RSI':<10} {'EMA RÁPIDA':<15} {'EMA LENTA':<15} {'ADX':<10} {'MOTIVO'}"
    print(header)
    print("-" * 110)
    
    for op in candidatos_para_display:
        sym = op["symbol"]
        acao = op["acao"]
        score = op["score"]
        d = op["data"]
        motivo = op["motivo"]

        # --- Lógica de Cores para Indicadores ---
        
        # 1. EMA (Direção da Tendência)
        ema_f_color = VERMELHO
        if d.get("EMA_FAST") is not None and d.get("EMA_SLOW") is not None:
             if (acao == "COMPRA" and d["EMA_FAST"] > d["EMA_SLOW"]) or \
                (acao == "VENDA" and d["EMA_FAST"] < d["EMA_SLOW"]):
                 ema_f_color = VERDE
            
        # 2. RSI (Condição de Sobrevenda/Sobrecompra)
        rsi_color = VERMELHO
        if d.get("RSI") is not None:
            if (acao == "COMPRA" and d["RSI"] < pc["rsi_max"]) and (d["RSI"] < 50):
                rsi_color = VERDE
            elif (acao == "VENDA" and d["RSI"] > pv["rsi_min"]) and (d["RSI"] > 50):
                rsi_color = VERDE

        # 3. ADX (Força da Tendência)
        adx_color = VERMELHO
        if d.get("ADX") is not None:
            if d["ADX"] > config.ADX_MIN_THRESHOLD:
                adx_color = VERDE

        # --- Formatação dos Dados ---
        rsi_val = d.get('RSI', 0.0)
        ema_f_val = d.get('EMA_FAST', 0.0)
        ema_s_val = d.get('EMA_SLOW', 0.0)
        adx_val = d.get('ADX', 0.0)

        rsi_formatted = f"{rsi_color}{rsi_val:.1f}{RESET}"
        ema_f_formatted = f"{ema_f_color}{ema_f_val:.2f}{RESET}"
        ema_s_formatted = f"{ema_s_val:.2f}"
        adx_formatted = f"{adx_color}{adx_val:.1f}{RESET}"
        score_formatted = f"{score:.1f}"

        # Linha formatada com as cores
        line = f"{sym:<10} {acao:<8} {score_formatted:<8} {rsi_formatted:<10} {ema_f_formatted:<15} {ema_s_formatted:<15} {adx_formatted:<10} {motivo}"
        print(line)
    
    print("-" * 110)
    
    # 8. EXECUÇÃO DE ORDENS
    for op in oportunidades_para_execucao:
        # Checa P/L do dia antes de entrar
        _, pl_pct = utils.get_daily_profit_loss()
        if pl_pct < config.DAILY_LOSS_LIMIT_PCT:
            logger.warning("Stop Loss Diário atingido. Pausando novas entradas.")
            break
            
        executar_ordem(op)

def run():
    if not mt5.initialize():
        print("Falha no MT5 Init")
        return
    
    # Inicializa a thread de trailing stop
    t = threading.Thread(target=utils.trailing_stop_thread, daemon=True)
    t.start()
    
    try:
        while True:
            # Controle via Arquivo (Headless Friendly)
            if os.path.exists("STOP.CMD"):
                logger.info("Arquivo STOP.CMD detectado. Encerrando...")
                break
                
            if utils.is_trading_time():
                # O ciclo principal agora gerencia a limpeza e a impressão da tela
                ciclo_principal() 
                time.sleep(config.CHECK_INTERVAL_SLOW)
            else:
                print(f"Aguardando Mercado... {datetime.now().time()}", end='\r')
                time.sleep(60)
                
    except KeyboardInterrupt:
        logger.info("Interrupção Manual.")
    finally:
        utils.trailing_running = False
        mt5.shutdown()

if __name__ == "__main__":
    run()