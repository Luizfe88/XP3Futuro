# utils.py – MÓDULO INSTITUCIONAL 2026

import logging, csv, os, time, winsound, json, random, math, numpy as np, pandas as pd, pandas_ta as ta
import MetaTrader5 as mt5
from datetime import datetime, timedelta
from collections import deque
import pickle
from colorama import init, Fore, Style
import config 

# ==================== CONFIGURAÇÃO DE CORES E LOG ====================
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
    fh.setFormatter(fmt); ch.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(ch)

# Variável global utilizada pelo bot.py para rastreio
EQUITY_DROP_HISTORY = deque(maxlen=2)

# ==================== CONEXÃO E SERVIÇOS MT5 ====================
def check_mt5_connection():
    """Conecta ou checa a conexão com o MetaTrader5."""
    if not mt5.initialize():
        logger.error(f"{VERMELHO}Falha ao inicializar MT5. Erro: {mt5.last_error()}{RESET}")
        return False
    
    if not mt5.account_info():
        logger.error(f"{VERMELHO}Conectado ao MT5, mas falha ao obter informações da conta. {mt5.last_error()}{RESET}")
        return False
        
    logger.info(f"{VERDE}Conexão MT5 OK! Conta: {mt5.account_info().login} | Servidor: {mt5.account_info().server}{RESET}")
    return True

def trailing_stop_service():
    """Serviço de Trailing Stop executado em thread separada (Apenas monitora, não é Soft Stop)."""
    while True:
        try:
            positions = mt5.positions_get()
            if positions is None:
                time.sleep(1)
                continue
                
            for pos in positions:
                # Lógica simplificada
                pass 
                
        except Exception as e:
            logger.error(f"{VERMELHO}Erro no serviço de Trailing Stop: {e}{RESET}")
            
        time.sleep(config.CHECK_INTERVAL_SLOW * 0.5)

# ==================== GESTÃO DE RISCO E PNL ====================

def get_daily_profit_loss():
    """
    Retorna PnL flutuante (realizado + flutuante) do dia atual em Reais e Porcentagem.
    """
    acc_info = mt5.account_info()
    if acc_info is None:
        return 0.0, 0.0
    
    pnl_reais = acc_info.profit 
    
    if acc_info.balance > 0:
        pnl_pct = pnl_reais / acc_info.balance 
    else:
        pnl_pct = 0.0
        
    return pnl_reais, pnl_pct * 100 

# ==================== FUNÇÃO MODIFICADA: SOFT STOP UNIFICADO ====================
def check_circuit_breakers(acc_info, tick_data, daily_start_equity):
    """
    Verifica se o limite de VaR Diário (Soft Stop 1) ou o Drawdown Diário (Soft Stop 2) foi atingido.
    Retorna True se qualquer um dos limites foi violado (CB_ACTIVE).
    """
    
    # 1. Checagem do VaR Diário (Soft Stop 1) - Baseado na perda flutuante
    current_loss_abs = acc_info.profit * -1 
    var_limit_abs = acc_info.equity * config.VAR_95_DAILY_LIMIT
    
    if current_loss_abs >= var_limit_abs:
        logger.critical(f"{VERMELHO}🚨 SOFT STOP (VaR) ATIVADO: Perda flutuante (R$ {current_loss_abs:,.2f}) excedeu VaR Limite ({config.VAR_95_DAILY_LIMIT*100:.1f}%)! Novas operações suspensas.{RESET}")
        return True
        
    # 2. Checagem do Drawdown Diário (Soft Stop 2) - Baseado no Equity
    if daily_start_equity > 0:
        current_drawdown_pct = (daily_start_equity - acc_info.equity) / daily_start_equity
        
        if current_drawdown_pct >= config.MAX_DAILY_DRAWDOWN_PCT:
            logger.critical(f"{VERMELHO}🚨 SOFT STOP (Drawdown) ATIVADO: Drawdown Diário ({current_drawdown_pct*100:.2f}%) excedeu limite ({config.MAX_DAILY_DRAWDOWN_PCT*100:.1f}%)! Novas operações suspensas.{RESET}")
            return True
            
    return False # Nenhum limite violado
# ==================== FIM DA FUNÇÃO MODIFICADA ====================

# ==================== FILTRO DE LIQUIDEZ ====================
def is_liquid_asset(symbol):
    """Checa se o ativo atende aos critérios mínimos de liquidez (ADV e Volume Ratio)."""
    return True # Assume True para a demonstração

# ==================== REGIME DE MERCADO ====================

# Funções auxiliares para o regime ML (Omitidas por brevidade)
def extract_market_features():
    return [0.5, 0.3, 25.0]

def predict_regime_ml():
    try:
        with open(config.ML_MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        features = extract_market_features()
        pred = model.predict([features])[0]
        mapping = {0: "CRISIS", 1: "BEAR", 2: "SIDEWAYS", 3: "BULL", 4: "STRONG_BULL"}
        return mapping.get(pred, "SIDEWAYS")
    except: 
        return "SIDEWAYS"

def get_market_regime():
    """Determina o Regime de Mercado (ML ou Heurístico)."""
    if config.USE_ML_REGIME and os.path.exists(config.ML_MODEL_PATH):
        regime_str = predict_regime_ml()
    else:
        regime_str = "BULL"
        
    return regime_str, 130000.0, 125000.0, 0.25 

# ==================== POSITION SIZING FINAL ====================\
def calcular_tamanho_posicao(symbol, sl_price, is_buy):
    """
    Calcula o lote ideal baseado em Risco Fixo (1% do Equity) e Stop Loss.
    """
    acc = mt5.account_info()
    equity = acc.equity
    tick = mt5.symbol_info_tick(symbol)
    
    if acc is None or tick is None or not tick.last:
        return 0.0

    current_price = tick.last
    risco_por_trade = equity * 0.01 
    
    if sl_price == 0.0:
        distancia_sl = 1.0
    else:
        distancia_sl = abs(current_price - sl_price)

    if distancia_sl < 0.01:
         logger.warning(f"Distância do SL para {symbol} muito pequena ({distancia_sl:.4f}). Retornando lote 0.")
         return 0.0

    lote_bruto = risco_por_trade / distancia_sl
    lote_final = (math.floor(lote_bruto / 100)) * 100
    
    if lote_final < 100:
        lote_final = 0
        
    return lote_final

# --- Adicionar ao utils.py ---

def check_trade_signal(df, params, side):
    """
    Verifica se o ativo atende aos critérios de COMPRA ou VENDA, dependendo do 'side'.
    Retorna: (True/False, string_status_completo)
    """
    if df is None or df.empty or len(df) < max(params['ema_slow'], 14): # Checagem de dados mínimos
        return False, "Dados insuficientes (DF Vazio)"

    # Certifica-se de que a última linha tem os indicadores
    last_row = df.iloc[-1]
    
    # 1. Extração de Valores
    ema_f_val = last_row[f'EMA_{params["ema_fast"]}']
    ema_s_val = last_row[f'EMA_{params["ema_slow"]}']
    rsi_val = last_row['RSI_14']
    adx_val = last_row['ADX_14']
    mom_val = last_row['MOMENTUM'] 
    
    # 2. Definição das Condições
    # NOTA: O critério de falha será usado para o status do relatório
    
    # --- Condição EMA ---
    if side == "COMPRA":
        cond_ema = ema_f_val > ema_s_val
        status_ema = f"F{ema_f_val:.2f}/L{ema_s_val:.2f}"
    else: # VENDA (EMA Rápida deve estar ABAIXO da Lenta)
        cond_ema = ema_f_val < ema_s_val
        status_ema = f"F{ema_f_val:.2f}/L{ema_s_val:.2f}"

    # --- Condição RSI ---
    rsi_level = params['rsi_level']
    if side == "COMPRA":
        cond_rsi = rsi_val >= rsi_level
    else: # VENDA (RSI deve ser igual ou ABAIXO do nível de VENDA, ex: 40)
        # Assumimos que o RSI_LEVEL no params_bear.json é o limite superior de VENDA (ex: 40)
        cond_rsi = rsi_val <= rsi_level

    # --- Condição ADX ---
    cond_adx = adx_val >= params['adx_min']

    # --- Condição Momentum ---
    mom_min_pct = params['momentum_min_pct']
    if side == "COMPRA":
        cond_mom = mom_val >= mom_min_pct
        mom_check = f"{mom_val:.2f}% (> {mom_min_pct:.2f}%)"
    else: # VENDA (Momentum deve ser negativo e mais forte que o limite invertido)
        # Ex: Se params['momentum_min_pct'] for 0.40, a condição é mom_val <= -0.40
        mom_target = mom_min_pct * -1.0 
        cond_mom = mom_val <= mom_target
        mom_check = f"{mom_val:.2f}% (< {mom_target:.2f}%)"


    # 3. Status Final e Mensagem
    all_conds = cond_ema and cond_rsi and cond_adx and cond_mom
    
    status = ""
    if not cond_ema: status = "Falha EMA"
    elif not cond_rsi: status = "Falha RSI"
    elif not cond_adx: status = "Falha ADX"
    elif not cond_mom: status = "Falha Momentum"

    final_status = f"{VERDE}APROVADO{RESET}" if all_conds else f"{VERMELHO}{status}{RESET}"
    
    # Retorno completo para o relatório
    report_status = {
        'ema': status_ema,
        'rsi': f"{rsi_val:.2f} (> {rsi_level})" if side == "COMPRA" else f"{rsi_val:.2f} (< {rsi_level})",
        'adx': f"{adx_val:.2f} (> {params['adx_min']})",
        'mom': mom_check,
        'final': final_status
    }
    
    return all_conds, report_status

def prepare_data_for_scan(symbol, params, lookback_days=300): # <-- AGORA ACEITA 'params'
    """Baixa dados diários (D1) e calcula os indicadores necessários, usando os parâmetros de EMA."""
    
    try:
        df = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_D1, 0, lookback_days)
        if df is None or len(df) == 0:
            return None
        
        df = pd.DataFrame(df)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        df.rename(columns={'close': 'Close', 'open': 'Open', 'high': 'High', 'low': 'Low'}, inplace=True)
        
    except Exception as e:
        logger.error(f"Erro ao baixar dados D1 para {symbol}: {e}")
        return None

    # 2. Calcula Indicadores USANDO OS PARÂMETROS CORRETOS
    df.ta.ema(length=params['ema_fast'], append=True)
    df.ta.ema(length=params['ema_slow'], append=True)
    
    df.ta.rsi(length=14, append=True)
    df.ta.adx(length=14, append=True)
    
    df['MOMENTUM'] = (df['Close'].pct_change(periods=5)) * 100
    
    df.dropna(inplace=True)
    return df

def check_trade_signal(df, params, side):
    """
    Verifica se o ativo atende aos critérios de COMPRA ou VENDA, dependendo do 'side'.
    Retorna: (True/False, dict_status_completo)
    """
    # 1. Checa se os dados e os indicadores estão completos
    ema_fast_key = f'EMA_{params["ema_fast"]}'
    ema_slow_key = f'EMA_{params["ema_slow"]}'
    
    if df is None or df.empty or ema_fast_key not in df.columns:
        return False, {"EMA": "-", "RSI": "-", "ADX": "-", "MOM": "-", "STATUS": "Dados Inválidos"}

    last_row = df.iloc[-1]
    
    # 2. Extração de Valores
    ema_f_val = last_row[ema_fast_key]
    ema_s_val = last_row[ema_slow_key]
    rsi_val = last_row.get('RSI_14', 0)
    adx_val = last_row.get('ADX_14', 0)
    mom_val = last_row.get('MOMENTUM', 0)
    
    # 3. Lógica Condicional (Invertida para VENDA)
    rsi_level = params['rsi_level']
    mom_min_pct = params['momentum_min_pct']

    if side == "COMPRA":
        cond_ema = ema_f_val > ema_s_val
        cond_rsi = rsi_val >= rsi_level
        cond_mom = mom_val >= mom_min_pct
        mom_check = f"{mom_val:.2f}% (> {mom_min_pct:.2f}%)"
    else: # VENDA
        cond_ema = ema_f_val < ema_s_val # EMA Rápida ABAIXO da Lenta
        cond_rsi = rsi_val <= rsi_level  # RSI ABAIXO do limite
        mom_target = mom_min_pct * -1.0 
        cond_mom = mom_val <= mom_target # Momentum negativo e forte
        mom_check = f"{mom_val:.2f}% (< {mom_target:.2f}%)"

    cond_adx = adx_val >= params['adx_min'] # Condição ADX é a mesma

    # 4. Status Final e Mensagem
    all_conds = cond_ema and cond_rsi and cond_adx and cond_mom
    
    status_msg = ""
    if not cond_ema: status_msg = "Falha EMA"
    elif not cond_rsi: status_msg = "Falha RSI"
    elif not cond_adx: status_msg = "Falha ADX"
    elif not cond_mom: status_msg = "Falha Momentum"

    final_status = f"{VERDE}APROVADO{RESET}" if all_conds else f"{VERMELHO}{status_msg}{RESET}"
    
    # Retorno completo para o relatório
    report_status = {
        'EMA': f"F{ema_f_val:.2f}/L{ema_s_val:.2f}",
        'RSI': f"{rsi_val:.2f} ({'>' if side == 'COMPRA' else '<'} {rsi_level:.0f})",
        'ADX': f"{adx_val:.2f} (> {params['adx_min']:.0f})",
        'MOM': mom_check,
        'STATUS': final_status,
        'SINAL_COMPLETO': all_conds # Chave essencial para o sorting
    }
    
    return all_conds, report_status

def scanner_full_report(resultados_scan_detalhado, top_n=20):
    """Gera o relatório detalhado do scan para COMPRA e VENDA."""
    
    # NOVO: FILTRAGEM DE SEGURANÇA
    # Garante que todos os elementos são tuplas de 3 e o 3º elemento (detalhes) é um dicionário
    valid_results = [
        r for r in resultados_scan_detalhado 
        if isinstance(r, tuple) and len(r) == 3 and isinstance(r[2], dict)
    ]
    
    if not valid_results:
        return "\n== DECOMPOSIÇÃO DO SCAN ===\n-----------------------------------------------------------------------------------------------------------------------------------\nNENHUM RESULTADO VÁLIDO PARA EXIBIÇÃO.\n-----------------------------------------------------------------------------------------------------------------------------------\n"
        
    # 1. Filtra os aprovados e pega os top N (para ter uma amostra representativa)
    
    # Ordena: Aprovados primeiro, depois falhas
    valid_results.sort(key=lambda x: 0 if x[2].get('SINAL_COMPLETO') else 1)
    
    # Pega apenas os top_n para exibir
    top_results = valid_results[:top_n]
    
    report = "\n"
    report += f"== DECOMPOSIÇÃO DO SCAN (TOP {len(top_results)} SINAIS) ===\n"
    report += "-----------------------------------------------------------------------------------------------------------------------------------\n"
    report += f"{'SÍMBOLO':<10}{'LADO':<10}{'EMA (F/L)':<20}{'RSI (Atual/Nível)':<20}{'ADX (Atual/Min)':<20}{'MOMENTUM (Atual/Min)':<25}{'STATUS':<15}\n"
    report += "-----------------------------------------------------------------------------------------------------------------------------------\n"
    
    for symbol, side, detalhes in top_results:
        # Acessa com .get() para evitar KeyErrors no display
        line = f"{symbol:<10}"
        line += f"{detalhes.get('SIDE', side):<10}"
        line += f"{detalhes.get('EMA', '-'):<20}" 
        line += f"{detalhes.get('RSI', '-'):<20}"
        line += f"{detalhes.get('ADX', '-'):<20}"
        line += f"{detalhes.get('MOM', '-'):<25}"
        line += f"{detalhes.get('STATUS', 'N/A')}\n"
        report += line
        
    report += "-----------------------------------------------------------------------------------------------------------------------------------\n"
    
    return report