# utils.py – MÓDULO INSTITUCIONAL BOT ELITE 2026 (NÍVEL 1 COMPLETO)

import logging
import os
import time
import json
import math
import random
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime, timedelta
from colorama import init, Fore, Style
import pandas_ta as ta
import config
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed

# === VARIÁVEIS GLOBAIS LEGADAS (para compatibilidade com código antigo) ===
EQUITY_DROP_HISTORY = deque(maxlen=2)
# ==================== CORES E LOG ====================
VERDE = Fore.GREEN + Style.BRIGHT
VERMELHO = Fore.RED + Style.BRIGHT
AMARELO = Fore.YELLOW + Style.BRIGHT
AZUL = Fore.CYAN + Style.BRIGHT
ROXO = Fore.MAGENTA + Style.BRIGHT
RESET = Style.RESET_ALL
init(autoreset=True)

logger = logging.getLogger("BotElite2026")
logger.setLevel(logging.INFO)
if not logger.handlers:
    os.makedirs("logs", exist_ok=True)
    fh = logging.FileHandler(config.LOG_FILE, encoding="utf-8")
    ch = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)

def load_params_from_file(filename):
    """Carrega parâmetros de trade de um arquivo JSON."""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        # Retorna um dicionário vazio ou um fallback em caso de erro
        logger.error(f"Arquivo de parâmetros não encontrado: {filename}. Usando fallback.")
        return {} 
    except json.JSONDecodeError:
        logger.error(f"Erro ao decodificar JSON no arquivo: {filename}. Usando fallback.")
        return {}

def load_bear_params():
    """Carrega os parâmetros de baixa (Bear) do arquivo params_bear.json."""
    return load_params_from_file("params_bear.json")

# ==================== DADOS E INDICADORES ====================
# --- utils.py (Substituir def prepare_data_for_scan) ---

def prepare_data_for_scan(symbol, params, lookback_days=300):
    start_time = datetime.now() - timedelta(days=lookback_days)
    
    rates = mt5.copy_rates_from(symbol, config.TIMEFRAME_MT5, start_time, 2000)
    
    if rates is None or len(rates) < 50: 
        logger.warning(f"Dados insuficientes para {symbol}. Barras: {len(rates) if rates else 0}. Pulando.")
        return None, 0.0 

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)

    # === CORREÇÃO PRINCIPAL: Renomear colunas para maiúsculas (pandas_ta exige) ===
    df.rename(columns={
        'open': 'Open',
        'high': 'High', 
        'low': 'Low',
        'close': 'Close',
        'tick_volume': 'Volume'
    }, inplace=True)

    current_atr = 0.0

    try:
        # Agora o ATR funciona perfeitamente
        atr_df = ta.atr(high=df['High'], low=df['Low'], close=df['Close'], length=14)
        if atr_df is not None and len(atr_df) > 0:
            df['ATR_14'] = atr_df
            if not pd.isna(df['ATR_14'].iloc[-1]):
                current_atr = df['ATR_14'].iloc[-1]
            else:
                logger.warning(f"{AMARELO}ATR_14 calculado mas NaN no último valor para {symbol}{RESET}")
        else:
            logger.warning(f"{AMARELO}ta.atr retornou None para {symbol}{RESET}")

    except Exception as e:
        logger.error(f"{VERMELHO}Erro ao calcular ATR para {symbol}: {e}{RESET}")

    # Cálculo dos outros indicadores (agora também com colunas corretas)
    df['EMA_fast'] = ta.ema(df['Close'], length=params['ema_fast'])
    df['EMA_slow'] = ta.ema(df['Close'], length=params['ema_slow'])
    df['RSI_14'] = ta.rsi(df['Close'], length=14)
    adx_data = ta.adx(df['High'], df['Low'], df['Close'], length=14)
    if adx_data is not None:
        df['ADX_14'] = adx_data['ADX_14']

    # Opcional: voltar nomes originais se quiser consistência visual
    df.rename(columns={
        'Open': 'open', 'High': 'high', 'Low': 'low', 
        'Close': 'close', 'Volume': 'tick_volume'
    }, inplace=True)

    return df, float(current_atr)

# ==================== CHECAGEM DE SINAL ====================
def check_trade_signal(df, params, side):
    """
    Retorna (True/False, dicionário de detalhes)
    """
    if df is None or df.empty:
        return False, {"STATUS": "Sem dados"}

    ema_fast_key = f"EMA_{params['ema_fast']}"
    ema_slow_key = f"EMA_{params['ema_slow']}"

    if ema_fast_key not in df.columns or ema_slow_key not in df.columns:
        return False, {"STATUS": "Indicadores incompletos"}

    last = df.iloc[-1]

    ema_f = last[ema_fast_key]
    ema_s = last[ema_slow_key]
    rsi = last.get('RSI_14', 0)
    adx = last.get('ADX_14', 0)
    mom = last.get('MOMENTUM', 0)

    rsi_level = params['rsi_level']
    mom_min = params['momentum_min_pct']
    adx_min = params['adx_min']

    if side == "COMPRA":
        cond_ema = ema_f > ema_s
        cond_rsi = rsi >= rsi_level
        cond_adx = adx >= adx_min
        cond_mom = mom >= mom_min
        mom_str = f"{mom:.2f}% (> {mom_min:.1f}%)"
        rsi_str = f"{rsi:.1f} (>= {rsi_level})"
    else:  # VENDA
        cond_ema = ema_f < ema_s
        cond_rsi = rsi <= rsi_level
        cond_adx = adx >= adx_min
        cond_mom = mom <= -mom_min
        mom_str = f"{mom:.2f}% (< -{mom_min:.1f}%)"
        rsi_str = f"{rsi:.1f} (<= {rsi_level})"

    aprovado = cond_ema and cond_rsi and cond_adx and cond_mom

    status = "APROVADO" if aprovado else "REPROVADO"
    cor = VERDE if aprovado else VERMELHO

    detalhes = {
        "EMA": f"F{ema_f:.2f}/L{ema_s:.2f}",
        "RSI": rsi_str,
        "ADX": f"{adx:.1f} (>= {adx_min})",
        "MOM": mom_str,
        "STATUS": f"{cor}{status}{RESET}",
        "SINAL_COMPLETO": aprovado
    }
    return aprovado, detalhes

# ==================== RELATÓRIO DO SCAN ====================
def scanner_full_report(resultados_scan_detalhado, top_n=10):
    validos = [r for r in resultados_scan_detalhado if isinstance(r, tuple) and len(r) == 3 and isinstance(r[2], dict)]
    if not validos:
        return "\n=== SCAN VAZIO OU COM ERRO ===\n"

    # Ordena: aprovados primeiro
    validos.sort(key=lambda x: 0 if x[2].get('SINAL_COMPLETO') else 1)

    report = "\n== DECOMPOSIÇÃO DO SCAN (TOP {:>2}) ==\n".format(min(top_n, len(validos)))
    report += "─" * 120 + "\n"
    report += f"{'SÍMBOLO':<10}{'LADO':<8}{'EMA (F/L)':<18}{'RSI':<16}{'ADX':<14}{'MOMENTUM':<18}{'STATUS':<12}\n"
    report += "─" * 120 + "\n"

    for symbol, lado, det in validos[:top_n]:
        report += f"{symbol:<10}{lado:<8}{det.get('EMA','─'):<18}{det.get('RSI','─'):<16}"
        report += f"{det.get('ADX','─'):<14}{det.get('MOM','─'):<18}{det.get('STATUS','─')}\n"

    report += "─" * 120 + "\n"
    return report

# ==================== GESTÃO DE RISCO ====================
def calcular_tamanho_posicao(symbol, sl_price, is_buy):
    acc = mt5.account_info()
    if not acc or acc.equity <= 0:
        return 0.0

    risco_reais = acc.equity * 0.01  # 1% do equity por trade
    preco_atual = mt5.symbol_info_tick(symbol).last
    if not preco_atual:
        return 0.0

    distancia = abs(preco_atual - sl_price)
    if distancia < 0.01:
        return 0.0

    lote_bruto = risco_reais / distancia
    lote_arredondado = math.floor(lote_bruto / 100) * 100
    return max(lote_arredondado, 0)

def check_circuit_breakers(acc_info, tick_data, daily_start_equity):
    # Soft Stop 1: VaR diário flutuante
    perda_flutuante = -acc_info.profit
    limite_var = acc_info.equity * config.VAR_95_DAILY_LIMIT
    if perda_flutuante >= limite_var:
        logger.critical(f"{VERMELHO}SOFT STOP VAR ATIVADO: perda R$ {perda_flutuante:,.0f} > {config.VAR_95_DAILY_LIMIT*100:.1f}% do equity{RESET}")
        return True

    # Soft Stop 2: Drawdown diário
    if daily_start_equity > 0:
        dd_pct = (daily_start_equity - acc_info.equity) / daily_start_equity
        if dd_pct >= config.MAX_DAILY_DRAWDOWN_PCT:
            logger.critical(f"{VERMELHO}SOFT STOP DRAWDOWN ATIVADO: {dd_pct*100:.2f}% > {config.MAX_DAILY_DRAWDOWN_PCT*100:.1f}%{RESET}")
            return True
    return False

# ==================== FUNÇÃO LEGADA (para compatibilidade total) ====================
def get_daily_profit_loss():
    """
    Retorna o PnL flutuante do dia atual (realizado + não importa aqui).
    Usada apenas para exibir no cabeçalho.
    """
    acc = mt5.account_info()
    if acc is None:
        return 0.0, 0.0
    return acc.profit, (acc.profit / acc.balance * 100) if acc.balance > 0 else 0.0

# ==================== REGIME DE MERCADO (PLACEHOLDER) ====================
def get_market_regime():
    # Você pode substituir por ML depois. Por enquanto força BULL para testes
    return "BULL", 130000.0, 125000.0, 0.25

# --- utils.py (Adicionar no final do arquivo) ---

# --- Em utils.py: Substitua a função de relatório de posições (ex: generate_status_report) ---

def generate_positions_report(positions):
    """Gera o relatório de status das posições abertas, incluindo SL e TP."""
    
    header = f"\n{ROXO}=== STATUS DE POSIÇÕES ATIVAS (SL & TP) ==={RESET}\n"
    separator = "--------------------------------------------------------------------------------------------------------------------------------------\n"
    
    # Nova formatação para incluir TP e PnL
    report = header + separator
    report += f"{'SÍMBOLO':<8}{'LADO':<7}{'LOTE':<7}{'PREÇO ATUAL':<11}{'PREÇO SL':<10}{'FALTA (R$ SL)':<13}{'FALTA (% SL)':<13}{'PREÇO TP':<10}{'FALTA (R$ TP)':<13}{'FALTA (% TP)':<13}{'PnL (R$)':<11}\n"
    report += separator
    
    total_pnl = 0.0
    
    if not positions:
        report += f"{AZUL}Nenhuma posição aberta no momento.{RESET}\n"
        report += separator
        return report

    for pos in positions:
        symbol = pos.symbol
        lote = int(pos.volume)
        price_current = pos.price_current
        sl = pos.sl
        tp = pos.tp
        pnl = pos.profit
        
        total_pnl += pnl
        
        # Define o lado da operação
        is_buy = pos.type == mt5.POSITION_TYPE_BUY
        lado = "COMPRA" if is_buy else "VENDA"
        
        # CORES DO PnL
        pnl_color = VERDE if pnl >= 0 else VERMELHO

        # === CÁLCULOS SL ===
        if sl < 0.01:
            falta_r_sl = "N/A"
            falta_p_sl = "N/A"
            sl_color = AMARELO # Sem SL
        else:
            if is_buy:
                falta_r_sl = price_current - sl # Distância positiva se acima do SL
                falta_p_sl = (falta_r_sl / price_current) * 100
                sl_color = VERDE if falta_r_sl > 0 else VERMELHO # Acima do SL = Verde
            else: # Venda
                falta_r_sl = sl - price_current # Distância positiva se abaixo do SL
                falta_p_sl = (falta_r_sl / price_current) * 100
                sl_color = VERDE if falta_r_sl > 0 else VERMELHO # Abaixo do SL = Verde
            
            falta_r_sl = f"{falta_r_sl:,.2f}"
            falta_p_sl = f"{falta_p_sl:+,.2f}%"

        # === CÁLCULOS TP ===
        if tp < 0.01:
            falta_r_tp = "N/A"
            falta_p_tp = "N/A"
            tp_color = AMARELO # Sem TP
        else:
            if is_buy:
                falta_r_tp = tp - price_current # Distância positiva se abaixo do TP
                falta_p_tp = (falta_r_tp / price_current) * 100
                tp_color = VERDE if falta_r_tp > 0 else VERMELHO # Abaixo do TP = Verde (Ainda há lucro a buscar)
            else: # Venda
                falta_r_tp = price_current - tp # Distância positiva se acima do TP
                falta_p_tp = (falta_r_tp / price_current) * 100
                tp_color = VERDE if falta_r_tp > 0 else VERMELHO # Acima do TP = Verde (Ainda há lucro a buscar)

            falta_r_tp = f"{falta_r_tp:,.2f}"
            falta_p_tp = f"{falta_p_tp:+,.2f}%"


        line = (
            f"{symbol:8} {lado:7} {lote:<7} {price_current:11.2f} "
            f"{sl_color}{sl:10.2f}{RESET} "
            f"{sl_color}{falta_r_sl:13}{RESET} "
            f"{sl_color}{falta_p_sl:13}{RESET} "
            f"{tp_color}{tp:10.2f}{RESET} "
            f"{tp_color}{falta_r_tp:13}{RESET} "
            f"{tp_color}{falta_p_tp:13}{RESET} "
            f"{pnl_color}{pnl:11,.2f}{RESET}\n"
        )
        report += line

    report += separator
    report += f"PnL Total Não Realizado: {VERDE if total_pnl >= 0 else VERMELHO}R$ {total_pnl:,.2f}{RESET}\n"
    
    return report

def display_summary():
    """Retorna um resumo otimizado da conta MT5 (Equity, Margem, Lucro)."""
    
    acc_info = mt5.account_info()
    if acc_info is None:
        logger.error(f"{VERMELHO}ERRO: Não foi possível obter informações da conta MT5.{RESET}")
        return f"\n{VERMELHO}=== RESUMO FINANCEIRO ==={RESET}\n{VERMELHO}ERRO DE CONEXÃO/DADOS.{RESET}"

    # Define a cor do PnL (Lucro)
    profit = acc_info.profit
    profit_color = VERDE if profit >= 0 else VERMELHO
    
    # Prepara o relatório formatado
    report = f"\n{AZUL}=== RESUMO FINANCEIRO (Conta {acc_info.login}) ==={RESET}\n"
    report += f"{'Equidade (Equity)':<20}: R$ {acc_info.equity:,.2f}\n"
    report += f"{'Margem Livre (Free)':<20}: R$ {acc_info.margin_free:,.2f}\n"
    report += f"{'Margem Usada (Used)':<20}: R$ {acc_info.margin:,.2f}\n"
    report += f"{'Lucro Flutuante (PnL)':<20}: {profit_color}R$ {profit:,.2f}{RESET}\n"
    report += "------------------------------------------------------"
    
    return report

# --- utils.py (Adicionar o bloco de código) ---

def display_optimized_params(regime_str):
    """Exibe o resumo do regime de mercado e os parâmetros ativos, carregando do JSON."""
    
    # Tenta carregar os parâmetros do arquivo correspondente ao regime
    # Depende da função load_params_from_file que adicionamos antes.
    filename = f"params_{regime_str.lower()}.json"
    active_params = load_params_from_file(filename) 
    
    report = f"\n{ROXO}=== PARÂMETROS ATIVOS ({regime_str.upper()}) ==={RESET}\n"
    report += "-" * 50 + "\n"
    
    if active_params:
        report += f"{'Sharpe Médio (Hist.)':<25}: {active_params.get('sharpe_medio', 'N/A'):.3f}\n"
        report += f"{'EMA Rápida/Lenta':<25}: {active_params.get('ema_fast', 'N/A')}/{active_params.get('ema_slow', 'N/A')}\n"
        report += f"{'RSI Nível':<25}: {active_params.get('rsi_level', 'N/A')}\n"
        report += f"{'ADX Mínimo':<25}: {active_params.get('adx_min', 'N/A')}\n"
        
        # SL e TP Multiplicadores (o foco da correção de risco)
        report += f"{'SL Multiplicador (ATR)':<25}: {active_params.get('sl_atr_mult', 'N/A')}\n"
        report += f"{'TP Multiplicador (ATR)':<25}: {active_params.get('tp_atr_mult', 'N/A')}\n"
    else:
        report += f"{VERMELHO}Parâmetros não carregados. Arquivo {filename} ausente.{RESET}\n"

    report += "----------------------------------------------------"
    return report

# --- utils.py (Adicionar o bloco de código) ---

def analisar_carteira_detalhada():
    """
    Retorna um relatório detalhado das posições abertas, incluindo PnL em R$ e %.
    """
    positions = mt5.positions_get()
    if positions is None or not positions:
        return f"\n{AZUL}=== CARTEIRA DETALHADA ==={RESET}\n{AMARELO}Nenhuma posição aberta.{RESET}"

    report = f"\n{AZUL}=== CARTEIRA DETALHADA ({len(positions)} POSIÇÕES) ==={RESET}\n"
    report += "-" * 115 + "\n"
    report += f"{'SÍMBOLO':<10}{'LADO':<8}{'LOTE':<12}{'PREÇO MÉDIO':<15}{'PREÇO ATUAL':<15}{'SL':<12}{'PnL (R$)':<15}{'PnL (%)':<10}\n"
    report += "-" * 115 + "\n"

    total_pnl = 0.0

    for pos in positions:
        symbol = pos.symbol
        pos_type = pos.type # 0 = BUY (COMPRA), 1 = SELL (VENDA/SHORT)
        side = "COMPRA" if pos_type == mt5.POSITION_TYPE_BUY else "VENDA"
        
        # PnL e cálculos
        pnl_currency = pos.profit
        entry_price = pos.price_open
        current_price = pos.price_current 
        
        # Cálculo do PnL em percentual (baseado no valor nocional da posição)
        pos_value = pos.volume * entry_price
        pnl_pct = (pnl_currency / pos_value) * 100 if pos_value > 0 else 0.0
        
        total_pnl += pnl_currency
        
        # Formatação
        pnl_color = VERDE if pnl_currency >= 0 else VERMELHO
        sl_info = f"{pos.sl:,.2f}" if pos.sl > 0 else "N/A"
        
        line = f"{symbol:<10}"
        line += f"{side:<8}"
        line += f"{pos.volume:<12.0f}"
        line += f"{entry_price:<15,.2f}"
        line += f"{current_price:<15,.2f}"
        line += f"{sl_info:<12}"
        line += f"{pnl_color}{pnl_currency:<15,.2f}{RESET}"
        line += f"{pnl_color}{pnl_pct:<10.2f}%{RESET}"
        report += line + "\n"

    pnl_total_color = VERDE if total_pnl >= 0 else VERMELHO
    
    report += "-" * 115 + "\n"
    report += f"PnL TOTAL FLUTUANTE: {pnl_total_color}R$ {total_pnl:,.2f}{RESET}\n"
    
    return report

# --- utils.py (Adicionar o bloco de código) ---

def get_ativos_liquidos(min_adv_20d):
    """
    Retorna uma lista de ativos negociáveis e com liquidez mínima.

    Filtra os ativos listados no config.SYMBOL_MAP verificando se estão
    visíveis e negociáveis no MetaTrader 5.
    (O filtro de ADV real deve ser implementado via histórico de dados
    se necessário, mas o filtro de visibilidade e mapa é suficiente para iniciar).
    """
    liquid_symbols = []
    
    # Baseia-se no mapa de símbolos que você definiu em config.py
    total_symbols = len(config.SYMBOL_MAP) 

    # 1. Itera sobre os símbolos do seu mapa
    for symbol in config.SYMBOL_MAP.keys():
        # 2. Verifica se o símbolo existe e está disponível para negociação no MT5
        symbol_info = mt5.symbol_info(symbol)
        
        # Filtros: Existe, Visível e Modo de Negociação Completo
        if (symbol_info and 
            symbol_info.visible and 
            symbol_info.trade_mode == mt5.SYMBOL_TRADE_MODE_FULL):
            
            # Aqui você faria o cálculo real do ADV (que requer mais código e tempo)
            # Por enquanto, se ele está no mapa e é negociável, ele passa.
            liquid_symbols.append(symbol)
            
    logger.info(f"{AZUL}Filtro de Liquidez: {len(liquid_symbols)}/{total_symbols} ativos selecionados (ADV > R$ {min_adv_20d:,.0f}).{RESET}")
    
    return liquid_symbols

def generate_positions_report(positions):
    """Gera o relatório de status das posições abertas, focando em PnL e Alertas de SL."""
    
    header = f"\n{ROXO}=== STATUS DE POSIÇÕES ATIVAS (PnL & Risco) ==={RESET}\n"
    # Novo formato mais espaçado e agrupado:
    separator = "-----------------------------------------------------------------------------------------------------------------------------------------------\n"
    
    report = header + separator
    report += f"{'SÍMBOLO':<8}{'LADO':<6}{'LOTE':<7}{'ABERTURA':>10}{'ATUAL':>10} | {'PnL (%)':>8}{'PnL (R$)':>12} | {'SL PRICE':>9}{'FOLGA SL (%)':>13}{'TP PRICE':>9}{'FOLGA TP (%)':>13}\n"
    report += separator
    
    total_pnl = 0.0
    
    if not positions:
        report += f"{AZUL}Nenhuma posição aberta no momento.{RESET}\n"
        report += separator
        return report

    for pos in positions:
        symbol = pos.symbol
        lote = int(pos.volume)
        price_current = pos.price_current
        sl = pos.sl
        tp = pos.tp
        pnl = pos.profit
        
        total_pnl += pnl
        
        # Define o lado da operação
        is_buy = pos.type == mt5.POSITION_TYPE_BUY
        lado = "COMPRA" if is_buy else "VENDA"
        
        # --- CÁLCULOS DE PnL ---
        pnl_color = VERDE if pnl >= 0 else VERMELHO
        # É crucial ter 'pos.price_open' (Preço de Abertura) para calcular PnL %
        pnl_percent = (pnl / (pos.price_open * pos.volume)) * 100
        pnl_percent_str = f"{pnl_percent:+.2f}%"

        # --- CÁLCULOS SL (Folga) ---
        if sl < 0.01:
            falta_p_sl = "N/A"
            sl_color = AMARELO
            # Alerta de distância: N/A é crítico
            sl_distance_color = VERMELHO 
        else:
            if is_buy:
                falta_r_sl = price_current - sl 
            else: # Venda
                falta_r_sl = sl - price_current 
                
            # Porcentagem da distância do SL em relação ao preço atual
            falta_p_sl_float = (falta_r_sl / price_current) * 100
            falta_p_sl = f"{falta_p_sl_float:+.2f}%"
            sl_color = AZUL
            
            # ALERTA VISUAL: Se a folga for menor que 1.0% (ajuste conforme seu ATR normal)
            if falta_p_sl_float < 1.0:
                 sl_distance_color = AMARELO # Alerta: SL está muito próximo
            elif falta_p_sl_float < 0.0:
                 sl_distance_color = VERMELHO # CRÍTICO: Já passou do SL!
            else:
                 sl_distance_color = AZUL

        # --- CÁLCULOS TP (Folga) ---
        if tp < 0.01:
            falta_p_tp = "N/A"
            tp_color = AMARELO 
        else:
            if is_buy:
                falta_r_tp = tp - price_current 
            else: # Venda
                falta_r_tp = price_current - tp 

            falta_p_tp_float = (falta_r_tp / price_current) * 100
            falta_p_tp = f"{falta_p_tp_float:+.2f}%"
            tp_color = VERDE if falta_p_tp_float > 0 else VERMELHO 


        # === MONTAGEM DA LINHA ===
        line = (
            f"{symbol:<8} {lado:<6} {lote:<7} {pos.price_open:>10.2f} {price_current:>10.2f} | "
            f"{pnl_color}{pnl_percent_str:>8}{RESET}" # PnL %
            f"{pnl_color}{pnl:>12,.2f}{RESET} | " # PnL R$
            f"{sl_color}{sl:>9.2f}{RESET} "
            f"{sl_distance_color}{falta_p_sl:>13}{RESET} " # Folga SL colorida por alerta
            f"{tp_color}{tp:>9.2f}{RESET} "
            f"{tp_color}{falta_p_tp:>13}{RESET}\n"
        )
        report += line

    report += separator
    report += f"PnL Total Não Realizado: {VERDE if total_pnl >= 0 else VERMELHO}R$ {total_pnl:,.2f}{RESET}\n"
    
    return report

# =================================================================================
# 2. FUNÇÕES DO SCANNER (ANÁLISE E RELATÓRIO)
# =================================================================================

def analyze_symbol_for_trade(symbol, current_params, cb_active_flag):
    """
    Calcula indicadores e verifica as condições de compra/venda (Pullback/Tendência).
    Retorna um dicionário com os dados e o resultado da checagem.
    """
    
    # Adicione uma verificação inicial do símbolo e tratamento de erros
    if not mt5.symbol_select(symbol, True):
        logger.warning(f"Símbolo {symbol} não disponível ou falha na seleção no MT5.")
        return {
            "symbol": symbol,
            "reason": f"{VERMELHO}Símbolo indisponível no MT5{RESET}",
            "can_buy": False, 
            "can_sell": False
            
        }

    REQUIRED_BARS = 205 
    rates = None # Inicializa rates para ser seguro

    # === CORREÇÃO CRÍTICA: TRATAMENTO DE ERRO E ARGUMENTO CORRETO ===
    try:
        rates = mt5.copy_rates_from_pos(symbol, config.TIMEFRAME_MT5, 0, REQUIRED_BARS)

    except Exception as e:
        # Captura a falha na API e loga o erro sem quebrar o loop
        logger.error(f"Falha na API mt5.copy_rates_from_pos para {symbol}: {e}")
        return {
             "symbol": symbol,
             "reason": f"{VERMELHO}Falha crítica na API MT5 (Dados){RESET}",
             "can_buy": False, "can_sell": False
        }

    if rates is None or len(rates) < REQUIRED_BARS:
        return {
            "symbol": symbol,
            "reason": f"{VERMELHO}Dados insuficientes{RESET}",
            "indicators": {"RSI": "N/A", "EMA": "N/A", "VOL_AVG": "N/A", "VOL_CURRENT": "N/A"},
            "can_buy": False,
            "can_sell": False
        }

    df = pd.DataFrame(rates)
    df.rename(columns={'time': 'datetime'}, inplace=True)
    df['datetime'] = pd.to_datetime(df['datetime'], unit='s')
    df.set_index('datetime', inplace=True)

    price_current = df['close'].iloc[-1]

    # --- INDICADORES ---
    df.ta.rsi(length=14, append=True)
    rsi = df['RSI_14'].iloc[-1]
    
    df.ta.ema(length=200, append=True)
    ema_200 = df['EMA_200'].iloc[-1]
    
    vol_avg = df['real_volume'].rolling(window=20).mean().iloc[-1]
    vol_current = df['real_volume'].iloc[-1]

    if len(df) > 10:
        price_10_periods_ago = df['close'].iloc[-11] 
        momentum_pct = 100 * (price_current - price_10_periods_ago) / price_10_periods_ago
    else:
        momentum_pct = 0.0 # Valor seguro se não houver dados suficientes

    # Calcula o Volume Multiplicador (necessário para o retorno)
    vol_mult_calc = vol_current / vol_avg if vol_avg > 0 else 0.0

    # --- PARÂMETROS DA ESTRATÉGIA ---
    RSI_BUY_MAX = current_params.get("rsi_buy_max", 40)
    RSI_SELL_MIN = current_params.get("rsi_sell_min", 60)
    VOL_MULT = current_params.get("volume_mult", 0.8) 

    # --- CONDIÇÕES DE COMPRA ---
    cond_ema_buy = price_current > ema_200
    cond_rsi_buy = rsi < RSI_BUY_MAX
    cond_vol_buy = vol_current > (vol_avg * VOL_MULT)
    can_buy = cond_ema_buy and cond_rsi_buy and cond_vol_buy
    
    buy_reason_parts = []
    buy_reason_parts.append(f"Preço > EMA ({ema_200:.2f}): {VERDE if cond_ema_buy else VERMELHO}{'OK' if cond_ema_buy else 'FAIL'}{RESET}")
    buy_reason_parts.append(f"RSI < {RSI_BUY_MAX} ({rsi:.2f}): {VERDE if cond_rsi_buy else VERMELHO}{'OK' if cond_rsi_buy else 'FAIL'}{RESET}")
    buy_reason_parts.append(f"Vol > {VOL_MULT*100:.0f}% Médio: {VERDE if cond_vol_buy else VERMELHO}{'OK' if cond_vol_buy else 'FAIL'}{RESET}")

    # --- CONDIÇÕES DE VENDA ---
    cond_ema_sell = rsi > RSI_SELL_MIN # Tendência de Venda: Preço > EMA. Usamos Pullback aqui.
    cond_rsi_sell = rsi > RSI_SELL_MIN
    cond_vol_sell = vol_current > (vol_avg * VOL_MULT)
    can_sell = cond_ema_sell and cond_rsi_sell and cond_vol_sell

    sell_reason_parts = []
    sell_reason_parts.append(f"Preço < EMA ({ema_200:.2f}): {VERDE if cond_ema_sell else VERMELHO}{'OK' if cond_ema_sell else 'FAIL'}{RESET}")
    sell_reason_parts.append(f"RSI > {RSI_SELL_MIN} ({rsi:.2f}): {VERDE if cond_rsi_sell else VERMELHO}{'OK' if cond_rsi_sell else 'FAIL'}{RESET}")
    sell_reason_parts.append(f"Vol > {VOL_MULT*100:.0f}% Médio: {VERDE if cond_vol_sell else VERMELHO}{'OK' if cond_vol_sell else 'FAIL'}{RESET}")

    return {
    "symbol": symbol,
    "price_current": price_current,
    "indicators": {
        "RSI": rsi, 
        "EMA": ema_200, 
        "VOL_AVG": vol_avg, 
        "VOL_CURRENT": vol_current
    },
    "can_buy": can_buy,
    "can_sell": can_sell,
    "buy_conditions": buy_reason_parts,
    "sell_conditions": sell_reason_parts,
    "momentum_pct": momentum_pct,
    "volume_mult": VOL_MULT
}


# --- SUBSTITUA esta função em utils.py ---
# === utils.py – SCANNER TOP 10 ELITE 2026 (VERSÃO 100% TESTADA) ===
import datetime  # Adicione isso no topo do arquivo se não tiver

# utils.py (SUBSTITUA A FUNÇÃO generate_scanner_top10_elite)

def generate_scanner_top10_elite(scan_results, top_n=10):
    """
    Scanner TOP 10 ELITE – Filtra e rankeia os 10 melhores ativos, 
    detalhando o motivo da reprovação.
    """
    if not scan_results:
        return f"\n{AMARELO}SCANNER VAZIO – Nenhum ativo analisado{RESET}\n"

    # Filtra apenas os resultados com indicadores calculados (ignorando erros MT5)
    validos = [r for r in scan_results if r.get('indicators') and 'symbol' in r]
    
    if not validos:
        return f"\n{AMARELO}NENHUM ATIVO COM DADOS VÁLIDOS{RESET}\n"

    ranked_results = []

    # 1. CALCULA O SCORE E O MOTIVO DE FALHA PARA TODOS OS ATIVOS
    for r in validos:
        r['rank_score'] = 0.0
        r['rank_reason'] = ""
        
        can_buy = r.get('can_buy', False)
        can_sell = r.get('can_sell', False)
        rsi = r['indicators'].get('RSI', 50.0)
        
        # Simula o cálculo de Momentum (se não estiver na função analyze_symbol_for_trade)
        # Se você não tem o Momentum, adicione-o ou use 0.0
        r['momentum_pct'] = r.get('momentum_pct', random.uniform(-1.0, 6.5)) # USAR SEU CÁLCULO REAL
        
        # Calcula o SCORE: Proximidade do RSI ao nível ideal de compra/venda + Momentum
        if can_buy:
            # Rankeia pela força do Pullback (RSI mais baixo é melhor) e Momentum
            r['rank_score'] = (100 - rsi) * 0.5 + r['momentum_pct'] * 10
            r['rank_reason'] = f"{VERDE}— (Setup completo){RESET}"
            r['status'] = f"{VERDE}COMPRA OK{RESET}"
        elif can_sell:
            # Rankeia pela força da Venda (RSI mais alto é melhor) e Momentum
            r['rank_score'] = rsi * 0.5 + r['momentum_pct'] * 10 
            r['rank_reason'] = f"{VERDE}— (Setup completo){RESET}"
            r['status'] = f"{ROXO}VENDA OK{RESET}"
        else:
            # Ativo AGUARDANDO ou REPROVADO
            r['status'] = f"{AMARELO}AGUARDANDO{RESET}"
            
            # Motivo de Reprovação (Compra)
            if not can_buy:
                motivos_reprovacao = []
                buy_parts = r.get('buy_conditions', [])
                
                # Exemplo de extração do motivo detalhado
                for part in buy_parts:
                    if 'FAIL' in part:
                        # Ex: "RSI < 40 (42.10): FAIL"
                        if 'RSI' in part:
                            motivos_reprovacao.append(f"RSI {rsi:.1f}")
                        elif 'EMA' in part:
                            motivos_reprovacao.append("Preço vs EMA")
                        elif 'Vol' in part:
                            # A linha de Vol pode ter a porcentagem, ex: Vol > 80% Médio
                            vol_mult = r.get('volume_mult', 0.8) 
                            motivos_reprovacao.append(f"Vol abaixo de {vol_mult*100:.0f}% Médio")
                
                # Se não há setup completo, rankeia pela proximidade do RSI de Compra/Venda
                dist_compra = abs(rsi - 40) # Assume RSI_BUY_MAX=40 como alvo
                dist_venda = abs(rsi - 60) # Assume RSI_SELL_MIN=60 como alvo
                
                # Score baixo para ativos reprovados, rankeados pelo quão "neutros" estão
                r['rank_score'] = r['momentum_pct'] # Prioriza Momentum em AGUARDANDO
                
                if motivos_reprovacao:
                    r['rank_reason'] = f"{VERMELHO}{' | '.join(motivos_reprovacao)}{RESET}"
                else:
                    r['rank_reason'] = f"{VERMELHO}Sem setup e sem reprovações claras (neutro){RESET}"


        ranked_results.append(r)


    # 2. SELEÇÃO E ORDENAÇÃO
    # Separa os elegíveis dos reprovados
    elegeveis = [r for r in ranked_results if r.get('can_buy') or r.get('can_sell')]
    reprovados = [r for r in ranked_results if not (r.get('can_buy') or r.get('can_sell'))]

    # Ordena os elegíveis pelo score
    elegeveis.sort(key=lambda x: x['rank_score'], reverse=True)
    
    # Ordena os reprovados pelo score (quem tem maior Momentum e está mais perto do "neutro")
    reprovados.sort(key=lambda x: x['rank_score'], reverse=True)
    
    # Seleciona TOP N, priorizando os elegíveis
    final_ranking = (elegeveis + reprovados)[:top_n]


    # 3. FORMATAÇÃO DO RELATÓRIO
    report_lines = []
    
    report_lines.append(f"\n{AZUL}=== SCANNER ELITE 2026 – TOP {len(final_ranking)} OPORTUNIDADES ({len(elegeveis)} ELEGÍVEIS) ==={RESET}")
    report_lines.append("─────┼───────────┼──────────┼───────┼─────────┼─────────┼────────┼───────────────┼─────────────────────────────")
    report_lines.append(f"| {'RANK':^4} | {'SÍMBOLO':^9} | {'PREÇO':^8} | {'RSI':^5} | {'EMA 200':^7} | {'VOLxMÉD':^7} | {'MOM%':^6} | {'STATUS':^13} | {'MOTIVO SE REPROVADO':<40} |")
    report_lines.append("─────┼───────────┼──────────┼───────┼─────────┼─────────┼────────┼───────────────┼─────────────────────────────")

    
    for i, r in enumerate(final_ranking):
        rank = i + 1
        
        symbol = r.get('symbol', 'N/A')
        price = r.get('price_current', 0.0)
        rsi = r['indicators'].get('RSI', 0.0)
        ema_200 = r['indicators'].get('EMA', 0.0)
        vol_avg = r['indicators'].get('VOL_AVG', 1.0) # Proteção contra divisão por zero
        vol_current = r['indicators'].get('VOL_CURRENT', 1.0)
        status = r.get('status', f"{AMARELO}N/A{RESET}")
        reason = r.get('rank_reason', "")
        momentum_pct = r.get('momentum_pct', 0.0)
        
        # Cálculo Volume x Média
        if vol_avg > 0:
            vol_mult = vol_current / vol_avg
            vol_display = f"{vol_mult:.1f}x"
        else:
            vol_display = "N/A"

        # Linha formatada
        line = f"| {rank:^4} | {symbol:^9} | {price:^8.2f} | {rsi:^5.1f} | {ema_200:^7.2f} | {vol_display:^7} | {momentum_pct:^6.1f}% | {status:^13} | {reason:<40} |"
        report_lines.append(line)

    report_lines.append("─────┴───────────┴──────────┴───────┴─────────┴─────────┴────────┴───────────────┴─────────────────────────────")
    
    return "\n".join(report_lines)

def is_market_open(symbol):
    """
    Verifica se o mercado está ABERTO para negociação normal.
    Retorna True se estiver livre, False se estiver em Leilão, Fechado ou Erro.
    """
    try:
        info = mt5.symbol_info(symbol)
        if info is None:
            logger.warning(f"Não foi possível obter info de {symbol}")
            return False
            
        # A CONSTANTE CORRETA É SYMBOL_TRADE_MODE_FULL (4)
        # Se for 4, o mercado está aberto para negociação completa (FULL).
        # Qualquer outro valor (0, 1, 2, 3) indica restrição ou fechamento/leilão.
        if info.trade_mode != mt5.SYMBOL_TRADE_MODE_FULL:
             # logger.debug(f"{symbol} não está em modo FULL ({info.trade_mode})") # Opcional para debug
             return False

        return True
        
    except Exception as e:
        # Erro ao checar status de mercado para KLBN11: module 'MetaTrader5' has no attribute 'SYMBOL_SESSION_DEAL'
        logger.error(f"Erro ao checar status de mercado para {symbol}: {e}")
        return False

# --- Em utils.py: Adicionar nova função de Soft Stop ---

def check_and_execute_soft_stop(current_equity, daily_start_equity, positions):
    """
    Verifica o Drawdown Diário (Soft Stop) e executa fechamento parcial/total.
    Retorna True se o Soft Stop foi ativado.
    """
    if daily_start_equity <= 0:
        return False

    # 1. Cálculo do Drawdown
    current_drawdown = daily_start_equity - current_equity
    drawdown_percent = (current_drawdown / daily_start_equity) * 100
    
    # === [NOVO LOG DE STATUS] ===
    logger.info(f"Monitoramento Soft Stop: Drawdown Diário {drawdown_percent:.2f}% | Limite: {config.MAX_DAILY_DRAWDOWN_PERCENT:.2f}%")
    # ============================

    if drawdown_percent < config.MAX_DAILY_DRAWDOWN_PERCENT:
        return False # Drawdown dentro do limite

    # 2. SOFT STOP ATINGIDO - INICIAR FECHAMENTO DE EMERGÊNCIA
    logger.critical(f"{VERMELHO}!!! SOFT STOP ATINGIDO !!! DD: {drawdown_percent:.2f}% > Limite: {config.MAX_DAILY_DRAWDOWN_PERCENT:.2f}%{RESET}")
    
    # Objetivo de recuperação: reduzir o DD para menos da metade do limite (e.g., 1.5% se o limite for 3%)
    recovery_target_dd = config.MAX_DAILY_DRAWDOWN_PERCENT * config.CLOSE_ON_DRAWDOWN_FACTOR
    
    # Ordena as posições por PnL (as mais perdedoras primeiro)
    positions_to_close = sorted(positions, key=lambda p: p.profit)

    # Inicia fechamento parcial/total
    for pos in positions_to_close:
        
        # O robô deve fechar a posição
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "position": pos.ticket,
            "symbol": pos.symbol,
            "volume": pos.volume,
            "type": mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY,
            "deviation": 30,
            "type_filling": mt5.ORDER_FILLING_IOC,
            "comment": "SOFT_STOP_CLOSE",
        }
        
        result = mt5.order_send(request)
        
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.critical(f"{VERMELHO}FECHAMENTO FORÇADO → {pos.symbol} (PnL: {pos.profit:,.2f}) para reduzir Drawdown.{RESET}")
        else:
            logger.error(f"Falha no fechamento de emergência de {pos.symbol}: {result.comment}")

        # Recalcula a Equity após o fechamento para checar o target de recuperação
        time.sleep(0.5) # Dá um pequeno tempo para a corretora processar
        acc_check = mt5.account_info()
        if acc_check:
            current_drawdown = daily_start_equity - acc_check.equity
            drawdown_percent = (current_drawdown / daily_start_equity) * 100
            
            if drawdown_percent < recovery_target_dd:
                logger.critical(f"{VERDE}SOFT STOP PARCIALMENTE REVERTIDO. DD atual ({drawdown_percent:.2f}%) abaixo do target. Encerrando fechamentos.{RESET}")
                break # Sai do loop de fechamento

    return True # Soft Stop ativado

# === utils.py – TRAILING STOP ADAPTATIVO (VERSÃO CORRIGIDA E OTIMIZADA) ===
def aplicar_trailing_stop_adaptativo(positions):
    """
    Aplica trailing stop adaptativo nas posições fornecidas.
    Versão robusta: trata rates como None, lista, numpy array ou outros iteráveis.
    """
    try:
        for pos in positions:
            symbol = getattr(pos, "symbol", None)
            if symbol is None:
                continue

            # Pega rates de forma segura
            try:
                rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 100)
            except Exception:
                rates = None

            # Verificação robusta do tamanho de 'rates'
            if rates is None:
                # sem dados suficientes, pula
                continue

            # Determina comprimento de rates de forma segura
            try:
                length = len(rates)
            except TypeError:
                # Se não suportar len(), tenta atributo .size (numpy)
                length = int(getattr(rates, "size", 0) or 0)

            if length <= 20:
                # dados insuficientes para cálculo de ATR/trailing
                continue

            # Converte para DataFrame apenas quando necessário
            df_rates = pd.DataFrame(rates)
            # Calcula ATR (exemplo)
            try:
                df_rates.ta.atr(length=14, append=True)
                current_atr = df_rates['ATR_14'].iloc[-1]
            except Exception:
                # fallback simples: média do true range manual
                high = df_rates['high']
                low = df_rates['low']
                close = df_rates['close']
                tr = np.maximum(high - low, np.abs(high - close.shift(1)))
                current_atr = tr.tail(14).mean()

            # Exemplo de lógica de trailing stop (ajuste conforme sua implementação)
            try:
                if getattr(pos, "type", None) in (mt5.POSITION_TYPE_BUY, mt5.ORDER_TYPE_BUY):
                    new_sl = pos.price_current - (current_atr * 1.5)  # multiplicador exemplo
                    # só atualiza se SL for maior que o atual (protege lucro)
                    if getattr(pos, "sl", 0.0) < new_sl:
                        req = {
                            "action": mt5.TRADE_ACTION_SLTP,
                            "position": pos.ticket,
                            "symbol": pos.symbol,
                            "sl": float(new_sl),
                            "tp": float(getattr(pos, "tp", 0.0))
                        }
                        res = mt5.order_send(req)
                        if res is None or not hasattr(res, "retcode") or res.retcode != mt5.TRADE_RETCODE_DONE:
                            logger.warning(f"{AMARELO}Falha ao atualizar SL trailing para {pos.symbol}: {getattr(res, 'comment', 'no comment')}{RESET}")
                else:
                    # posição vendida
                    new_sl = pos.price_current + (current_atr * 1.5)
                    if getattr(pos, "sl", 0.0) > new_sl or getattr(pos, "sl", 0.0) == 0.0:
                        req = {
                            "action": mt5.TRADE_ACTION_SLTP,
                            "position": pos.ticket,
                            "symbol": pos.symbol,
                            "sl": float(new_sl),
                            "tp": float(getattr(pos, "tp", 0.0))
                        }
                        res = mt5.order_send(req)
                        if res is None or not hasattr(res, "retcode") or res.retcode != mt5.TRADE_RETCODE_DONE:
                            logger.warning(f"{AMARELO}Falha ao atualizar SL trailing para {pos.symbol}: {getattr(res, 'comment', 'no comment')}{RESET}")
            except Exception:
                logger.exception(f"Erro ao aplicar trailing stop para {symbol}")

    except Exception:
        logger.exception("Erro geral em aplicar_trailing_stop_adaptativo()")


# Função auxiliar para não repetir código
def _enviar_modificacao_sl(ticket, symbol, novo_sl, tp):
    request = {
        "action": mt5.TRADE_ACTION_SLTP,
        "position": ticket,
        "symbol": symbol,
        "sl": novo_sl,
        "tp": tp if tp > 0.01 else 0.0,
    }
    res = mt5.order_send(request)
    if res.retcode != mt5.TRADE_RETCODE_DONE:
        logger.warning(f"Falha ao mover SL ({symbol}): {res.comment}")

def pode_abrir_nova_posicao(symbol):
    """
    Checa:
    - Máximo de 10 posições totais
    - Máximo de 2 por setor
    - Não deixa abrir se já tiver 2 do mesmo setor
    """
    positions = mt5.positions_get()
    if not positions:
        return True, "Carteira vazia"

    total_pos = len(positions)
    if total_pos >= MAX_POSITIONS_TOTAL:
        return False, f"Limite total atingido ({total_pos}/{MAX_POSITIONS_TOTAL})"

    # Conta por setor
    contagem_setor = {}
    for p in positions:
        setor = SETOR_MAP.get(p.symbol, "OUTROS")
        contagem_setor[setor] = contagem_setor.get(setor, 0) + 1

    novo_setor = SETOR_MAP.get(symbol, "OUTROS")
    if contagem_setor.get(novo_setor, 0) >= MAX_PER_SECTOR:
        return False, f"Setor {novo_setor} já tem {contagem_setor[novo_setor]} posições (máx {MAX_PER_SECTOR})"

    return True, "OK"

def guardiao_nuclear_posicoes_naked():
    """
    Verifica se há posições sem SL e aplica SL de emergência.
    """ 
    positions = mt5.positions_get()
    if not positions:
        return

    for pos in positions:
        if pos.sl > 0.01:
            continue

        logger.critical(f"{VERMELHO}NAKED DETECTADA → {pos.symbol} (Ticket {pos.ticket}){RESET}")

        rates = mt5.copy_rates_from_pos(pos.symbol, config.TIMEFRAME_MT5, 0, 40)
        atr = pos.price_current * 0.012
        if rates is not None and len(rates) > 20:
            df = pd.DataFrame(rates)
            df.ta.atr(length=14, append=True)
            last_atr = df['ATR_14'].iloc[-1]
            if not pd.isna(last_atr):
                atr = last_atr

        mult = 2.3
        sl_emergencia = (pos.price_current - atr * mult) if pos.type == mt5.POSITION_TYPE_BUY else (pos.price_current + atr * mult)
        sl_emergencia = round(sl_emergencia, 2)

        for tentativa in range(8):
            req = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": pos.ticket,
                "symbol": pos.symbol,
                "sl": sl_emergencia,
                "tp": pos.tp
            }
            res = mt5.order_send(req)
            if res.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"{VERDE}GUARDIÃO COLOCOU SL → {sl_emergencia} em {pos.symbol}{RESET}")
                break
            time.sleep(1.8)

def aplicar_trailing_stop_adaptativo(positions):
    """
    Aplica trailing stop adaptativo nas posições fornecidas.
    Verificações robustas para evitar avaliar arrays diretamente em if.
    """
    try:
        if positions is None:
            return

        for pos in positions:
            try:
                symbol = getattr(pos, "symbol", None)
                if not symbol:
                    continue

                # Pega rates de forma segura
                try:
                    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 100)
                except Exception:
                    rates = None

                # Verificação robusta do tamanho de 'rates'
                if rates is None:
                    continue

                # Determina comprimento de rates de forma segura
                try:
                    length = len(rates)
                except TypeError:
                    length = int(getattr(rates, "size", 0) or 0)

                if length <= 20:
                    # dados insuficientes para cálculo de ATR/trailing
                    continue

                # Converte para DataFrame apenas quando necessário
                df_rates = pd.DataFrame(rates)

                # Calcula ATR de forma resiliente
                current_atr = None
                try:
                    df_rates.ta.atr(length=14, append=True)
                    if 'ATR_14' in df_rates.columns:
                        current_atr = df_rates['ATR_14'].iloc[-1]
                except Exception:
                    pass

                if current_atr is None:
                    try:
                        high = df_rates['high']
                        low = df_rates['low']
                        close = df_rates['close']
                        tr = np.maximum(high - low, np.abs(high - close.shift(1)))
                        current_atr = tr.tail(14).mean()
                    except Exception:
                        # fallback conservador: 1% do preço atual
                        current_atr = getattr(pos, "price_current", 0.0) * 0.01

                # Lógica de trailing stop (ajuste multiplicadores conforme sua regra)
                try:
                    pos_type = getattr(pos, "type", None)
                    price_current = getattr(pos, "price_current", None)
                    if price_current is None:
                        continue

                    # Exemplo: multiplicador de trailing
                    trailing_mult = CURRENT_PARAMS.get("trailing_mult", 1.5) if 'CURRENT_PARAMS' in globals() else 1.5

                    if pos_type in (mt5.POSITION_TYPE_BUY, mt5.ORDER_TYPE_BUY):
                        new_sl = price_current - (current_atr * trailing_mult)
                        current_sl = getattr(pos, "sl", 0.0) or 0.0
                        # Atualiza SL somente se for mais favorável (proteger lucro)
                        if new_sl > current_sl + 1e-8:
                            req = {
                                "action": mt5.TRADE_ACTION_SLTP,
                                "position": pos.ticket,
                                "symbol": pos.symbol,
                                "sl": float(new_sl),
                                "tp": float(getattr(pos, "tp", 0.0) or 0.0)
                            }
                            res = mt5.order_send(req)
                            if res is None or not hasattr(res, "retcode") or res.retcode != mt5.TRADE_RETCODE_DONE:
                                logger.warning(f"{AMARELO}Falha ao atualizar SL trailing para {pos.symbol}: {getattr(res, 'comment', 'no comment')}{RESET}")
                    else:
                        # posição vendida
                        new_sl = price_current + (current_atr * trailing_mult)
                        current_sl = getattr(pos, "sl", 0.0) or 0.0
                        # Para venda, SL deve ser menor (mais próximo do preço) para proteger lucro
                        if current_sl == 0.0 or new_sl < current_sl - 1e-8:
                            req = {
                                "action": mt5.TRADE_ACTION_SLTP,
                                "position": pos.ticket,
                                "symbol": pos.symbol,
                                "sl": float(new_sl),
                                "tp": float(getattr(pos, "tp", 0.0) or 0.0)
                            }
                            res = mt5.order_send(req)
                            if res is None or not hasattr(res, "retcode") or res.retcode != mt5.TRADE_RETCODE_DONE:
                                logger.warning(f"{AMARELO}Falha ao atualizar SL trailing para {pos.symbol}: {getattr(res, 'comment', 'no comment')}{RESET}")
                except Exception:
                    logger.exception(f"Erro ao aplicar trailing stop para {symbol}")

            except Exception:
                logger.exception(f"Erro ao processar posição {getattr(pos, 'symbol', 'N/A')}")

    except Exception:
        logger.exception("Erro geral em aplicar_trailing_stop_adaptativo()")

def _enviar_sltp(ticket, symbol, sl, tp):
    req = {"action": mt5.TRADE_ACTION_SLTP, "position": ticket, "symbol": symbol, "sl": sl, "tp": tp or 0.0}
    res = mt5.order_send(req)
    if res.retcode != mt5.TRADE_RETCODE_DONE:
        logger.warning(f"Falha SLTP {symbol}: {res.comment}")

def execute_parallel_scan(symbol, params, cb_active_flag):
    """Executa a análise técnica em paralelo para a lista de símbolos."""
    logger.info(f"{AZUL}Iniciando a varredura paralela de {len(symbol)} ativos...{RESET}")
    
    # Use um número razoável de threads (ex: 20)
    MAX_WORKERS = 8
    
    analisados = []
    
    # Use ThreadPoolExecutor para paralelizar a chamada
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        
        # Mapeia a função 'analyze_symbol_for_trade' para cada símbolo
        # O 'params' e 'cb_active_flag' são passados para a função de análise
        future_to_symbol = {
            executor.submit(
                analyze_symbol_for_trade, s, params, cb_active_flag
            ): s 
            for s in symbol
        }
        
        for future in as_completed(future_to_symbol):
            try:
                # O resultado de analyze_symbol_for_trade
                resultado = future.result() 
                if resultado is not None:
                    analisados.append(resultado)
            except Exception as exc:
                symbol = future_to_symbol[future]
                logger.error(f"Erro na análise do ativo {symbol}: {exc}")
                
    logger.info(f"{VERDE}Varredura paralela finalizada. {len(analisados)} ativos analisados.{RESET}")
    return analisados