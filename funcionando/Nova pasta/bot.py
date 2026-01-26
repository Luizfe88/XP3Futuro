# bot.py – BOT ELITE 2026 XP EDITION – 100% FUNCIONAL NA XP (DEZ/2025)
import MetaTrader5 as mt5
import time
import keyboard
import threading
import os
import utils
import pandas
from concurrent.futures import ThreadPoolExecutor
import config
from utils import logger, is_trading_time, VERDE, VERMELHO, AMARELO, AZUL, ROXO, BRANCO, RESET, \
    get_asset_technical_data, tocar_som_compra, register_trade, close_all_positions, \
    get_open_positions_display, trailing_running, trailing_stop_thread
from colorama import init
from datetime import datetime, timedelta
import pandas as pd

init(autoreset=True)

last_candidates_data = []
LARG = 145
SPINNER = ['/', '—', '\\', '|']
spin_idx = 0

def selecionar_melhores_candidatos():
    logger.info("Carregando lista completa de candidatos")
    lista = [s + config.CORRETORA_SUFFIX for s in config.CANDIDATOS_BASE[:config.MAX_ASSETS_SCANEADOS]]
    logger.info(f"Lista dinâmica pronta → {len(lista)} ativos")
    return lista

def comprar(symbol):
    if not is_trading_time():
        return False, "Fora do horário"

    tick = mt5.symbol_info_tick(symbol)
    if not tick or tick.ask <= 0:
        return False, "Sem cotação"

    spread = tick.ask - tick.bid
    if spread > config.MAX_SPREAD_CENTS:
        return False, f"Spread alto {spread:.3f}"

    volume = utils.calcular_volume_inteligente(symbol)
    if volume < 100:
        return False, f"Volume insuficiente: {volume}"

    # COMPRA LIMPA (XP não aceita SL/TP na mesma ordem)
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": float(volume),
        "type": mt5.ORDER_TYPE_BUY,
        "price": tick.ask,
        "deviation": 50,
        "magic": 20251230,
        "comment": "Elite2026_XP",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    logger.info(f"{ROXO}ENVIANDO ORDEM → {symbol} | {volume} ações @ R${tick.ask:.5f}{RESET}")
    r = mt5.order_send(request)

    if r is None:
        logger.error(f"{VERMELHO}FALHA CRÍTICA → {symbol} | MT5 retornou None{RESET}")
        return False, "MT5 None"

    if r.retcode == mt5.TRADE_RETCODE_DONE:
        tocar_som_compra()
        logger.info(f"{VERDE}COMPRA EXECUTADA → {symbol} | {volume} ações | Ticket {r.order}{RESET}")
        register_trade(symbol, "BUY", tick.ask, volume, "OK")

        # SL/TP depois da abertura
        time.sleep(0.4)
        sl = round(tick.ask * (1 - config.STOP_LOSS_PCT), 2)
        tp = round(tick.ask * (1 + config.TAKE_PROFIT_PCT), 2)
        mt5.order_send({
            "action": mt5.TRADE_ACTION_SLTP,
            "position": r.order,
            "symbol": symbol,
            "sl": sl,
            "tp": tp,
        })
        logger.info(f"{AZUL}SL/TP aplicados: {sl} / {tp}{RESET}")
        config.SCREEN_UPDATE_REQUIRED = True
        return True, "OK"
    else:
        logger.error(f"{VERMELHO}REJEITADA → {symbol} | {r.retcode} - {r.comment}{RESET}")
        return False, r.comment

def scanner_ativos_detalhado(symbol):
    data, err = get_asset_technical_data(symbol)
    if err or not data:
        return {"symbol": symbol, "status": "ERRO", "reason": err[:30], "score": -1, "data": {}}

    d = data
    rejeicao = []

    if not d["tendencia_d1"]:
        rejeicao.append("Tendência D1 fraca")
    if not utils.ibov_ok():
        rejeicao.append("IBOV caindo")
    setor = utils.get_setor(symbol)
    if utils.contar_setor_atual().get(setor, 0) >= config.MAX_PER_SETOR:
        rejeicao.append(f"Setor {setor[:8]} cheio")
    if d["EMA_FAST"] <= d["EMA_SLOW"]:
        rejeicao.append("EMA3≤EMA15")
    if d["RSI"] >= config.RSI_OVERSOLD:
        rejeicao.append(f"RSI {d['RSI']:.0f}")
    if d["momentum"] < config.MIN_MOMENTUM_UP:
        rejeicao.append(f"Mom {d['momentum']:+.1f}%")
    vol_ratio = d["curr_vol"] / max(d["avg_vol"], 1)
    if vol_ratio < config.MIN_VOLUME_MULTIPLIER:
        rejeicao.append(f"Vol {vol_ratio:.1f}x")

    if rejeicao:
        return {"symbol": symbol, "status": "REJEITADO", "reason": " | ".join(rejeicao), "score": 0, "data": d}

    score = (0.4 * min(1, d["momentum"]/5) +
             0.3 * (1 - d["RSI"]/100) +
             0.3 * min(1, vol_ratio/3))

    return {"symbol": symbol, "status": "OPORTUNIDADE", "reason": f"R${d['price']:.2f}", "score": score, "data": d}

def executar_scan():
    global last_candidates_data
    with ThreadPoolExecutor(max_workers=25) as ex:
        resultados = list(ex.map(scanner_ativos_detalhado, config.CANDIDATOS_DINAMICOS))

    validos = [r for r in resultados if r["score"] > 0]
    validos.sort(key=lambda x: x["score"], reverse=True)

    rejeitados = [r for r in resultados if r["score"] <= 0 and r["status"] != "ERRO"]
    rejeitados.sort(key=lambda x: x["data"].get("curr_vol", 0), reverse=True)

    exibicao = validos[:20]
    faltam = 20 - len(exibicao)
    if faltam > 0:
        exibicao += rejeitados[:faltam]

    last_candidates_data = exibicao
    config.SCREEN_UPDATE_REQUIRED = True

    if validos:
        for cand in validos[:5]:
            ok, _ = comprar(cand["symbol"])
            if ok:
                break

def imprimir_tela(cands, next_scan):
    global spin_idx
    os.system('cls' if os.name == 'nt' else 'clear')
    ibov_status = "OK" if utils.ibov_ok() else "CAINDO"
    print(f"{BRANCO}BOT ELITE 2026 XP EDITION | B3 {'ABERTO' if is_trading_time() else 'FECHADO'} | IBOV {ibov_status} | {datetime.now().strftime('%H:%M:%S')}{RESET}\n")
    print("═" * 145)
    print(get_open_positions_display() or f"{AZUL}Nenhuma posição aberta{RESET}")
    print("═" * 145)
    print(f" TOP 20 ATIVOS SCANEADOS ".center(145, "═"))
    print(f" {'#':<3} {'ATIVO':<8} {'PREÇO':>10} {'SCORE':>7} {'RSI':>6} {'MOM%':>7} {'VOLx':>6} {'STATUS':<12} {'MOTIVO'}")
    print("─" * 145)

    for i, it in enumerate(cands):
        d = it.get("data", {})
        volx = d.get("curr_vol", 0) / max(d.get("avg_vol", 1), 1) if d else 0
        cor_preco = VERDE if d.get("price", 0) > 0 else VERMELHO
        cor_score = VERDE if it["score"] >= 0.7 else (AMARELO if it["score"] > 0 else VERMELHO)
        cor_rsi = VERDE if 30 <= d.get("RSI", 100) <= 65 else VERMELHO
        cor_mom = VERDE if d.get("momentum", 0) >= 1.0 else VERMELHO
        cor_vol = VERDE if volx >= 1.5 else VERMELHO
        cor_motivo = VERDE if it["status"] == "OPORTUNIDADE" else VERMELHO
        motivo = f"{cor_motivo}{it['reason'][:60]}{RESET}"

        print(f" {i+1:<3} {it['symbol']:<8} "
              f"{cor_preco}{d.get('price',0):10.3f}{RESET} "
              f"{cor_score}{it['score']:7.3f}{RESET} "
              f"{cor_rsi}{d.get('RSI',0):6.1f}{RESET} "
              f"{cor_mom}{d.get('momentum',0):+7.2f}{RESET} "
              f"{cor_vol}{volx:6.1f}x{RESET} "
              f"{it['status']:<12} {motivo}")

    spin_idx = (spin_idx + 1) % 4
    print("─" * 145)
    print(f" Próximo scan em {int(next_scan-time.time())}s {SPINNER[spin_idx]} | F9 = Zerar tudo ".center(145))
    print("═" * 145)

# Exemplo de como implementar em bot.py (necessita do pandas)
def gerar_relatorio_sumario():
    try:
        # Lê o CSV de trades
        df = pd.read_csv(config.CSV_FILE, names=["Time", "Symbol", "Side", "Price", "Volume", "Result"])
        df['Time'] = pd.to_datetime(df['Time'])
    except Exception as e:
        logger.error(f"Erro ao ler trades CSV: {e}")
        return

    # Filtra por trades nas últimas 30 minutos
    agora = datetime.now()
    ultimos_30m = agora - timedelta(minutes=30)
    df_recente = df[df['Time'] >= ultimos_30m]
    
    # Agregação de dados
    compras = df_recente[df_recente['Side'] == 'BUY']
    vendas = df_recente[df_recente['Side'].isin(['CLOSE_ALL', 'SELL'])] # Considera CLOSE_ALL como venda/saída

    total_comprado = compras['Volume'].sum()
    total_vendido = vendas['Volume'].sum()
    
    # Custo/Ganho (Simplificado - apenas para estimativa)
    # Custo (aprox): Preço * Volume
    # Custo na compra
    custo_compra = (compras['Price'].astype(float) * compras['Volume'].astype(int)).sum()
    
    # Receita na venda (simplificado, lucro real é mais complexo)
    receita_venda = (vendas['Price'].astype(float) * vendas['Volume'].astype(int)).sum()

    # O lucro/prejuízo real está em `pos.profit` no MT5 e no registro de TRADE
    # do bot, mas é mais complexo calcular a partir apenas do CSV de trades (sem o preço de entrada).
    
    logger.info(f"\n{AMARELO}*** RELATÓRIO SUMÁRIO (Últimos 30m - {agora.strftime('%H:%M:%S')}) ***{RESET}")
    logger.info(f"{AZUL}Ações Compradas:{RESET} {total_comprado} ações")
    logger.info(f"{AZUL}Ações Vendidas:{RESET} {total_vendido} ações")
    logger.info(f"{AZUL}Capital Gasto (Est.):{RESET} R${custo_compra:,.2f}")
    logger.info(f"{AZUL}Capital Ganho (Est.):{RESET} R${receita_venda:,.2f}")
    logger.info(f"{VERDE}Ações abertas:{RESET} {mt5.positions_total()}")
    logger.info("*"*50)

def run_bot():
    if not mt5.initialize():
        print("ERRO: Não foi possível conectar ao MetaTrader 5")
        return

    config.CANDIDATOS_DINAMICOS = selecionar_melhores_candidatos()
    utils.trailing_running = True
    threading.Thread(target=utils.trailing_stop_thread, daemon=True).start()

    logger.info("BOT ELITE 2026 XP EDITION INICIADO")
    next_scan = time.time()
    next_report = time.time() + 1800

    try:
        while True:
            if is_trading_time() and time.time() >= next_scan:
                executar_scan()
                next_scan = time.time() + config.CHECK_INTERVAL_SLOW

            if config.SCREEN_UPDATE_REQUIRED:
                imprimir_tela(last_candidates_data, next_scan)
                config.SCREEN_UPDATE_REQUIRED = False

            if is_trading_time() and time.time() >= next_scan:
                executar_scan()
                next_scan = time.time() + config.CHECK_INTERVAL_SLOW

            # Lógica do Relatório (NOVO)
            if time.time() >= next_report:
                gerar_relatorio_sumario() # Chamar a nova função
                next_report = time.time() + 1800 # Agendar próximo relatório em 30 min
                config.SCREEN_UPDATE_REQUIRED = True

            if keyboard.is_pressed('f9'):
                close_all_positions()
                config.SCREEN_UPDATE_REQUIRED = True
                time.sleep(1)

            time.sleep(0.3)
    except KeyboardInterrupt:
        print("\nBot encerrado.")
    finally:
        utils.trailing_running = False
        mt5.shutdown()

if __name__ == "__main__":
    run_bot()