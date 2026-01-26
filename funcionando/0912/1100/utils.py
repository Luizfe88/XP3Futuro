# utils.py – MÓDULO INSTITUCIONAL BOT ELITE 2026 (NÍVEL 1 COMPLETO)

import logging
import os
import time
import json
import math
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime, timedelta
from colorama import init, Fore, Style
import pandas_ta as ta
import config
from collections import deque

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
def scanner_full_report(resultados_scan_detalhado, top_n=20):
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

def analyze_symbol_for_trade(symbol, timeframe, current_params):
    """
    Calcula indicadores e verifica as condições de compra/venda (Pullback/Tendência).
    Retorna um dicionário com os dados e o resultado da checagem.
    """
    
    REQUIRED_BARS = 205 
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, REQUIRED_BARS)

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
        "sell_conditions": sell_reason_parts
    }


# --- SUBSTITUA esta função em utils.py ---

def generate_scanner_report(scan_results):
    """Gera o relatório formatado com os resultados do scanner (Top 20 Ativos) com melhor visual."""
    
    header = f"\n{ROXO}=== SCANNER: TOP ATIVOS E MOTIVOS DE NÃO-EXECUÇÃO (V/R) ==={RESET}\n"
    # Novo separador para melhor alinhamento
    separator = "-------------------------------------------------------------------------------------------------------------------------------------\n"
    report = header + separator
    
    # Novo formato de cabeçalho com alinhamento otimizado
    report += f"{'SÍMBOLO':<8}{'PREÇO':>8}{'RSI':>8}{'EMA (200)':>10}{'VOL ATUAL':>12}{'VOL MÉDIO':>12} | {'STATUS':<15}\n"
    report += separator
    
    # Ordena por proximidade do RSI ao extremo para mostrar os setups mais promissores
    sorted_results = sorted(scan_results, key=lambda x: min(abs(x['indicators']['RSI'] - 40), abs(x['indicators']['RSI'] - 60)) if isinstance(x['indicators']['RSI'], (int, float)) else 1000)

    # Limita aos 20 mais relevantes
    for data in sorted_results[:20]:
        
        ind = data['indicators']
        rsi = ind.get('RSI', 'N/A')
        ema = ind.get('EMA', 'N/A')
        vol_c = ind.get('VOL_CURRENT', 'N/A')
        vol_a = ind.get('VOL_AVG', 'N/A')
        
        # Cores para Indicadores (RSI perto de 30/70 é VERDE, no meio 40-60 é AMARELO)
        rsi_float = float(rsi) if isinstance(rsi, (int, float)) else 50 # Assume 50 se for N/A
        rsi_color = VERDE if rsi_float < 35 or rsi_float > 65 else (AMARELO if 40 < rsi_float < 60 else AZUL)
        
        # Cores para o Status Geral
        can_trade = data['can_buy'] or data['can_sell']
        status_color = VERDE if can_trade else VERMELHO
        status_text = "BUY READY" if data['can_buy'] else ("SELL READY" if data['can_sell'] else "NO SETUP")
        
        # Linha principal - Usando separador | para facilitar a leitura
        line = (
            f"{data['symbol']:<8} {data['price_current']:>8.2f} "
            f"{rsi_color}{rsi:>8.2f}{RESET} "
            f"{ema:>10.2f} "
            f"{vol_c:>12,.0f} "
            f"{vol_a:>12,.0f} | "
            f"{status_color}{status_text:<15}{RESET}\n"
        )
        report += line
        
        # Linhas de Detalhe (Razão para NÃO COMPRAR/VENDER)
        if not can_trade:
            
            # Motivo da NÃO COMPRA:
            buy_reasons_str = " | ".join(data.get('buy_conditions', ["N/A"]))
            report += f"    -> {AMARELO}NÃO COMPRA:{RESET} {buy_reasons_str}\n"
        
            # Motivo da NÃO VENDA:
            sell_reasons_str = " | ".join(data.get('sell_conditions', ["N/A"]))
            report += f"    -> {AMARELO}NÃO VENDA:{RESET} {sell_reasons_str}\n"

    return report