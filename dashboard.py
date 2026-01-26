# dashboard_v6.py - XP3 PRO B3 INSTITUTIONAL DASHBOARD v6.0
"""
🏛️ XP3 PRO B3 INSTITUTIONAL DASHBOARD - Nível Profissional
✅ Métricas avançadas de performance e risco
✅ Análise psicológica e disciplina
✅ Estatísticas por ativo, horário e estratégia
✅ Visualizações institucionais (Equity Curve, Drawdown, Heatmaps)
✅ Journal de Trading integrado
✅ Relatórios exportáveis em PDF
✅ Integração de dados de mercado em tempo real via MT5
"""

import os
# 🔇 SILENCIA LOGS DO TENSORFLOW
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import base64
import MetaTrader5 as mt5
from datetime import datetime, timedelta
import json
from pathlib import Path
import time
import sys
import os
import logging
import io
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

import re
# ✅ FORÇA UTF-8 NO WINDOWS (CRÍTICO PARA EMOJIS)
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        # Python < 3.7
        pass

# ===========================
# CONFIGURAÇÃO DE LOGGING
# ===========================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dashboard_institutional.log", encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# ===========================
# IMPORTAÇÕES DO BOT
# ===========================
try:
    if '.' not in sys.path:
        sys.path.insert(0, '.')
    
    import config
    import utils
    import news_filter
    import llm_narrative
    
    try:
        from bot import (
            bot_state, position_open_times, last_close_time, trading_paused,
            equity_inicio_dia, daily_max_equity, get_market_status, daily_trades_per_symbol
        )
        BOT_CONNECTED = True
    except ImportError as e:
        logger.warning(f"⚠️ Bot não conectado: {e}")
        BOT_CONNECTED = False
        
    try:
        from ml_optimizer import ml_optimizer
        ML_AVAILABLE = True
    except ImportError:
        ML_AVAILABLE = False
        
except ImportError as e:
    logger.error(f"❌ Erro crítico: {e}")
    st.error(f"❌ Erro ao importar módulos: {e}")
    st.stop()

# ===========================
# CONFIGURAÇÃO DA PÁGINA - TEMA INSTITUCIONAL
# ===========================
st.set_page_config(
    page_title="XP3 Institutional Dashboard",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS INSTITUCIONAL AVANÇADO
st.markdown("""
<style>
    /* Tema Principal - Clean & Professional */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Headers Hierárquicos */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a1a2e;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        border-bottom: 3px solid #0f3460;
        padding-bottom: 1rem;
    }
    
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #0f3460;
        margin: 1.5rem 0 1rem 0;
        border-left: 4px solid #16213e;
        padding-left: 1rem;
    }
    
    /* Cards Profissionais */
    .metric-card {
        background: #000000;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 4px solid #0f3460;
        transition: transform 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    
    .kpi-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: #000000;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .kpi-value {
        color: #000000;
        font-size: 2rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    .kpi-label {
        color: #000000;
        font-size: 0.9rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Status Badges */
    .status-badge {
        display: inline-block;
        padding: 0.4rem 1.2rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    .status-active {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
    }
    
    .status-warning {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    
    .status-danger {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        color: white;
    }
    
    /* Cores Semânticas */
    .profit-positive {
        color: #00b894;
        font-weight: 700;
    }
    
    .profit-negative {
        color: #d63031;
        font-weight: 700;
    }
    
    .profit-neutral {
        color: #636e72;
        font-weight: 600;
    }
    
    /* Alertas Estilizados */
    .alert-critical {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-weight: 600;
        border-left: 5px solid #c92a2a;
    }
    
    .alert-warning {
        background: linear-gradient(135deg, #ffa502 0%, #ff6348 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-weight: 600;
        border-left: 5px solid #e67e22;
    }
    
    .alert-info {
        background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
        color: #000000;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-weight: 600;
        border-left: 5px solid #2c3e50;
    }
    
    /* Tabelas Profissionais */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
    }
    
    .dataframe thead tr th {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        padding: 12px;
        text-align: left;
        font-weight: 600;
    }
    
    .dataframe tbody tr:hover {
        background-color: #f8f9fa;
    }
    
    /* Sidebar Melhorado */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f3460 0%, #16213e 100%);
        color: white;
    }
    
    section[data-testid="stSidebar"] .stMarkdown {
        color: white;
    }
    
    /* Botões Estilizados */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Melhorias nos Metrics */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
    }
    
    div[data-testid="stMetricDelta"] {
        font-size: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ===========================
# FUNÇÕES AUXILIARES AVANÇADAS
# ===========================

def load_execution_logs():
    """
    Lê e converte o log diário (TXT) em DataFrame para o Dashboard.
    Extrai: Hora, Ativo, Status, Motivo e Indicadores com Regex robusto.
    """
    log_dir = Path("analysis_logs")
    today = datetime.now().strftime("%Y-%m-%d")
    log_file = log_dir / f"analysis_log_{today}.txt"
    
    data = []
    if not log_file.exists():
        return pd.DataFrame()
        
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Encontra cada entrada começando com o emoji de relógio
        # O regex captura tudo desde 🕐 até o próximo 🕐 (ou fim do arquivo)
        entries = re.finditer(r"(🕐.*?)(?=🕐|$)", content, re.DOTALL)
        
        for match in entries:
            block = match.group(1).strip()
            
            # 1. Cabeçalho: 🕐 HH:MM:SS | SYMBOL | STATUS
            header_match = re.search(r"🕐\s*(\d{2}:\d{2}:\d{2})\s*\|\s*(\w+)\s*\|\s*(.+?)(?:\n|$)", block)
            if not header_match:
                continue
                
            timestamp = header_match.group(1)
            symbol = header_match.group(2)
            status = header_match.group(3).strip()
            
            # Limpeza de status (remove traços residuais causados por falta de newline no log antigo)
            status = re.split(r'-{2,}', status)[0].strip()
            
            # 2. Motivo: Captura após "Motivo:" até encontrar traços divisores ou fim
            # O (.*?) é lazy para não pegar os traços da próxima entrada se houver
            reason_match = re.search(r"💬 Motivo:\s*(.*?)(?=-{2,}|$)", block, re.DOTALL)
            reason = reason_match.group(1).strip() if reason_match else "N/A"
            
            # 3. Score: Tenta pegar "Score real", senão pega a porcentagem da barra de progresso
            score_match = re.search(r"Score real:\s*(\d+)", block)
            if score_match:
                score = score_match.group(1)
            else:
                # Fallback para o Log antigo que só tinha a barra de progresso
                setup_match = re.search(r"Filtros de Setup:.*?(\d+)%", block)
                score = setup_match.group(1) if setup_match else "0"
            
            # 4. Indicadores Chave
            rsi_match = re.search(r"• RSI:\s*([\d\.]+)", block)
            adx_match = re.search(r"• ADX:\s*([\d\.]+)", block)
            
            entry = {
                "Time": timestamp,
                "Symbol": symbol,
                "Status": status,
                "Reason": reason,
                "Score": int(score),
                "RSI": rsi_match.group(1) if rsi_match else "-",
                "ADX": adx_match.group(1) if adx_match else "-",
                "FullLog": block
            }
            data.append(entry)
                
    except Exception as e:
        logger.error(f"Erro ao fazer parse dos logs: {e}")
        
    if not data:
        return pd.DataFrame()
        
    # Mais recentes no topo
    return pd.DataFrame(data[::-1])
@st.cache_data(ttl=5)
def load_account_info():
    """Carrega informações da conta MT5"""
    try:
        if not mt5.initialize(path=config.MT5_TERMINAL_PATH):
            return None
        
        acc = mt5.account_info()
        if acc:
            return {
                "balance": acc.balance,
                "equity": acc.equity,
                "margin": acc.margin,
                "free_margin": acc.margin_free,
                "margin_level": (acc.equity / acc.margin * 100) if acc.margin > 0 else 0,
                "profit": acc.profit,
                "login": acc.login,
                "server": acc.server,
                "leverage": acc.leverage,
                "currency": acc.currency
            }
    except Exception as e:
        logger.error(f"Erro MT5: {e}")
    return None

@st.cache_data(ttl=5)
def load_positions():
    """Carrega posições abertas"""
    try:
        positions = mt5.positions_get() or []
        if not positions:
            return pd.DataFrame()
        
        data = []
        for pos in positions:
            side = "COMPRA" if pos.type == mt5.POSITION_TYPE_BUY else "VENDA"
            
            # Cálculo de tempo
            time_open = datetime.fromtimestamp(pos.time)
            duration = datetime.now() - time_open
            hours = int(duration.total_seconds() // 3600)
            minutes = int((duration.total_seconds() % 3600) // 60)
            time_str = f"{hours}h {minutes}m" if hours > 0 else f"{minutes}m"
            
            # P&L percentual
            if side == "COMPRA":
                pnl_pct = ((pos.price_current - pos.price_open) / pos.price_open) * 100
            else:
                pnl_pct = ((pos.price_open - pos.price_current) / pos.price_open) * 100
            
            data.append({
                "Ticket": pos.ticket,
                "Símbolo": pos.symbol,
                "Lado": side,
                "Volume": pos.volume,
                "Preço Entrada": pos.price_open,
                "Preço Atual": pos.price_current,
                "Stop Loss": pos.sl if pos.sl > 0 else 0,
                "Take Profit": pos.tp if pos.tp > 0 else 0,
                "P&L R$": pos.profit,
                "P&L %": pnl_pct,
                "Tempo Aberto": time_str,
                "Timestamp": pos.time
            })
        
        return pd.DataFrame(data)
    except Exception as e:
        logger.error(f"Erro ao carregar posições: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=30)
def load_historical_trades(days=30):
    """Carrega histórico de trades dos últimos N dias"""
    try:
        all_trades = []
        
        for i in range(days):
            date = datetime.now() - timedelta(days=i)
            filename = f"trades_log_{date.strftime('%Y-%m-%d')}.txt"
            
            if not os.path.exists(filename):
                continue
            
            with open(filename, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for line in lines[2:]:
                if line.strip() and not line.startswith('-'):
                    parts = line.split('|')
                    if len(parts) >= 8:
                        try:
                            timestamp_str = parts[0].strip()
                            tipo = parts[1].strip()
                            simbolo = parts[2].strip()
                            lado = parts[3].strip()
                            volume = float(parts[4].strip())
                            preco = float(parts[5].strip())
                            pnl_str = parts[6].strip().replace('R$', '').replace(',', '')
                            pnl = float(pnl_str) if pnl_str else 0
                            
                            all_trades.append({
                                'Data': date.strftime('%Y-%m-%d'),
                                'Timestamp': timestamp_str,
                                'Tipo': tipo,
                                'Símbolo': simbolo,
                                'Lado': lado,
                                'Volume': volume,
                                'Preço': preco,
                                'P&L': pnl,
                                'Hora': timestamp_str.split()[1] if len(timestamp_str.split()) > 1 else '00:00'
                            })
                        except:
                            continue
        
        return pd.DataFrame(all_trades)
    except Exception as e:
        logger.error(f"Erro ao carregar histórico: {e}")
        return pd.DataFrame()

def calculate_advanced_metrics(trades_df):
    """Calcula métricas avançadas de performance"""
    if trades_df.empty or 'P&L' not in trades_df.columns:
        return {}
    
    # Filtrar apenas trades fechados (SAÍDA)
    closed_trades = trades_df[trades_df['Tipo'] == 'SAÍDA'].copy()
    
    if closed_trades.empty:
        return {}
    
    returns = closed_trades['P&L'].values
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    
    total_trades = len(returns)
    winning_trades = len(wins)
    losing_trades = len(losses)
    
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = abs(losses.mean()) if len(losses) > 0 else 0
    
    profit_factor = (wins.sum() / abs(losses.sum())) if losses.sum() != 0 else 0
    
    # Expectativa matemática
    expectancy = (win_rate/100 * avg_win) - ((100-win_rate)/100 * avg_loss)
    
    # Sharpe Ratio (simplificado)
    if len(returns) > 1:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else 0
    else:
        sharpe = 0
    
    # Maximum Drawdown
    cumulative = np.cumsum(returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = cumulative - running_max
    max_dd = abs(drawdown.min()) if len(drawdown) > 0 else 0
    
    # Sortino Ratio
    negative_returns = returns[returns < 0]
    downside_std = negative_returns.std() if len(negative_returns) > 1 else 0
    sortino = (returns.mean() / downside_std) * np.sqrt(252) if downside_std > 0 else 0
    
    # Calmar Ratio (Anualizado / MaxDD)
    total_days = (trades_df['Timestamp'].max() - trades_df['Timestamp'].min()) / 86400 if len(trades_df) > 1 else 1
    total_days = max(total_days, 1)
    annualized_return = (returns.sum() / total_days) * 252
    calmar = (annualized_return / max_dd) if max_dd > 0 else 0
    
    # Maior sequência de ganhos/perdas
    win_streak = 0
    loss_streak = 0
    current_win_streak = 0
    current_loss_streak = 0
    
    for ret in returns:
        if ret > 0:
            current_win_streak += 1
            current_loss_streak = 0
            win_streak = max(win_streak, current_win_streak)
        elif ret < 0:
            current_loss_streak += 1
            current_win_streak = 0
            loss_streak = max(loss_streak, current_loss_streak)
    
    return {
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'expectancy': expectancy,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'calmar_ratio': calmar,
        'max_drawdown': max_dd,
        'win_streak': win_streak,
        'loss_streak': loss_streak,
        'total_pnl': returns.sum(),
        'avg_trade': returns.mean(),
        'best_trade': returns.max(),
        'worst_trade': returns.min()
    }

def get_performance_color(value, metric_type='profit'):
    """Retorna classe CSS baseada no valor"""
    if metric_type == 'profit':
        return 'profit-positive' if value >= 0 else 'profit-negative'
    elif metric_type == 'ratio':
        if value > 2:
            return 'profit-positive'
        elif value > 1:
            return 'profit-neutral'
        else:
            return 'profit-negative'
    elif metric_type == 'percentage':
        if value >= 60:
            return 'profit-positive'
        elif value >= 40:
            return 'profit-neutral'
        else:
            return 'profit-negative'

@st.cache_data(ttl=5)
def load_real_time_market_data(symbols=None):
    """Carrega dados de mercado em tempo real via MT5"""
    try:
        if not mt5.initialize(path=config.MT5_TERMINAL_PATH):
            return pd.DataFrame()
        
        if symbols is None:
            # Use elite symbols or default
            symbols = config.ELITE_SYMBOLS if hasattr(config, 'ELITE_SYMBOLS') else ['PETR4', 'VALE3', 'ITUB4', 'BBDC4']
        
        data = []
        for sym in symbols:
            tick = mt5.symbol_info_tick(sym)
            if tick:
                # Ajustar fuso horário para Brasília (UTC-3)
                time_brasilia = datetime.fromtimestamp(tick.time) + timedelta(hours=3)
                data.append({
                    'Símbolo': sym,
                    'Bid': tick.bid,
                    'Ask': tick.ask,
                    'Último': tick.last,
                    'Volume': tick.volume,
                    'Tempo': time_brasilia.strftime('%H:%M:%S')
                })
        
        mt5.shutdown()
        return pd.DataFrame(data)
    except Exception as e:
        logger.error(f"Erro ao carregar dados reais: {e}")
        return pd.DataFrame()

# ===========================
# HEADER PRINCIPAL
# ===========================
st.markdown('<div class="main-header">🏛️ XP3 INSTITUTIONAL DASHBOARD v6.0</div>', unsafe_allow_html=True)

# ===========================
# SIDEBAR - CONTROLES E FILTROS
# ===========================
with st.sidebar:
    st.image("https://via.placeholder.com/200x60/0f3460/FFFFFF?text=XP3+PRO", width='stretch')
    
    st.markdown("---")
    st.subheader("⚙️ Configurações")
    
    # Auto-refresh
    auto_refresh = st.checkbox("🔄 Atualização Automática", value=True)
    refresh_interval = st.slider("Intervalo (segundos)", 5, 120, 30)
    
    if st.button("🔄 Atualizar Agora", width='stretch'):
        st.cache_data.clear()
        st.rerun()
    
    st.markdown("---")
    st.subheader("📊 Período de Análise")
    
    analysis_period = st.selectbox(
        "Selecione o período",
        ["Hoje", "Últimos 7 dias", "Últimos 30 dias", "Este mês", "Personalizado"]
    )
    
    if analysis_period == "Personalizado":
        date_from = st.date_input("Data inicial", datetime.now() - timedelta(days=30))
        date_to = st.date_input("Data final", datetime.now())
    
    st.markdown("---")
    st.subheader("🎯 Filtros Avançados")
    
    show_only_active = st.checkbox("Apenas trades ativos", value=False)
    min_pnl_filter = st.number_input("P&L mínimo (R$)", value=0.0, step=10.0)
    
    st.markdown("---")
    st.subheader("📥 Exportação")
    
    if st.button("📄 Gerar Relatório PDF", width='stretch'):
        st.info("Funcionalidade em desenvolvimento")
    
    if st.button("📊 Exportar Excel Completo", width='stretch'):
        st.info("Funcionalidade em desenvolvimento")

# ===========================
# CARREGAMENTO DE DADOS
# ===========================
if "pause_refresh_until" not in st.session_state:
    st.session_state["pause_refresh_until"] = 0.0
acc = load_account_info()
positions_df = load_positions()
# Definir período baseado na seleção
if analysis_period == "Hoje":
    days_to_load = 1
elif analysis_period == "Últimos 7 dias":
    days_to_load = 7
elif analysis_period == "Últimos 30 dias":
    days_to_load = 30
else:
    days_to_load = 30
historical_trades = load_historical_trades(days=days_to_load)
if not acc:
    st.error("❌ Erro ao conectar com MT5. Verifique a configuração.")
    st.stop()

# ===========================
# SEÇÃO 1: PAINEL DE CONTROLE - KPI + STATUS
# ===========================
col_title, col_status = st.columns([0.65, 0.35])

with col_title:
    st.markdown('<div class="section-header">📊 Painel de Controle - Visão Geral</div>', unsafe_allow_html=True)

with col_status:
    try:
        status = get_market_status()
        regime = utils.detect_market_regime()
        now_str = datetime.now().strftime("%H:%M:%S")
        
        status_color = "#00b894" if status["trading_allowed"] else "#d63031"
        status_text = "ABERTO" if status["trading_allowed"] else "FECHADO"
        
        regime_color = "#00b894" if "BULL" in regime else "#d63031" if "BEAR" in regime else "#fdcb6e"
        
        st.markdown(
            f"""
            <div style="
                background-color: white; 
                padding: 10px; 
                border-radius: 8px; 
                border: 1px solid #ddd;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                text-align: right;
                font-family: 'Segoe UI', sans-serif;
            ">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;">
                    <span style="font-weight: bold; color: #1a1a2e; font-size: 0.9rem;">Mercado:</span>
                    <span style="font-weight: bold; color: {status_color}; background: {status_color}20; padding: 2px 8px; border-radius: 4px;">{status_text}</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;">
                    <span style="font-weight: bold; color: #1a1a2e; font-size: 0.9rem;">Regime:</span>
                    <span style="font-weight: bold; color: {regime_color}; background: {regime_color}20; padding: 2px 8px; border-radius: 4px;">{regime}</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;">
                    <span style="font-weight: bold; color: #1a1a2e; font-size: 0.9rem;">Horário:</span>
                    <span style="font-family: monospace; font-size: 1rem; color: #1a1a2e;">{now_str}</span>
                </div>
                 <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-weight: bold; color: #1a1a2e; font-size: 0.9rem;">Próx. Pregão:</span>
                    <span style="font-family: monospace; font-size: 0.9rem; color: #0984e3;">{status['countdown']}</span>
                </div>
            </div>
            """, 
            unsafe_allow_html=True
        )
    except Exception as e:
        st.error(f"Erro status: {e}")

# KPI Cards
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
with kpi1:
    st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
    st.markdown('<div class="kpi-label">💰 Patrimônio Líquido</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="kpi-value">R$ {acc["equity"]:,.2f}</div>', unsafe_allow_html=True)
    pnl_day = acc['profit']
    delta_color = "🟢" if pnl_day >= 0 else "🔴"
    st.markdown(f'<div style="font-size: 0.9rem;color: #000000">{delta_color} R$ {pnl_day:+,.2f} hoje</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with kpi2:
    st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
    st.markdown('<div class="kpi-label">📈 Retorno Diário</div>', unsafe_allow_html=True)
    pnl_pct = (acc['profit'] / acc['balance'] * 100) if acc['balance'] > 0 else 0
    st.markdown(f'<div class="kpi-value">{pnl_pct:+.2f}%</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="font-size: 0.9rem;color: #000000">Base: R$ {acc["balance"]:,.2f}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with kpi3:
    st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
    st.markdown('<div class="kpi-label">🎯 Posições Ativas</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="kpi-value">{len(positions_df)}</div>', unsafe_allow_html=True)
    max_pos = getattr(config, 'MAX_SYMBOLS', 10)
    st.markdown(f'<div style="font-size: 0.9rem;color: #000000">Limite: {max_pos} posições</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with kpi4:
    st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
    st.markdown('<div class="kpi-label">🛡️ Nível de Margem</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="kpi-value">{acc["margin_level"]:.1f}%</div>', unsafe_allow_html=True)
    margin_status = "Seguro" if acc['margin_level'] > 200 else "Atenção"
    st.markdown(f'<div style="font-size: 0.9rem;color: #000000">{margin_status}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with kpi5:
    lucro_flutuante = positions_df['P&L R$'].sum() if not positions_df.empty else 0
    st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
    st.markdown('<div class="kpi-label">💸 P&L Flutuante</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="kpi-value">R$ {lucro_flutuante:+,.2f}</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="font-size: 0.9rem;color: #000000">{len(positions_df)} operações abertas</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ===========================
# 🤖 IA & ELITE SYMBOLS (V5.2)
# ===========================
st.markdown("---")
ia_col1, ia_col2 = st.columns([1, 1])

with ia_col1:
    st.markdown('<div class="section-header">🤖 Narrativa de Mercado (IA)</div>', unsafe_allow_html=True)
    
    # Pega primeiro símbolo das posições ou um padrão
    active_sym = positions_df['Símbolo'].iloc[0] if not positions_df.empty else "IBOV"
    
    # Mock/Real indicators para o LLM
    dummy_ind = {"rsi": 45, "adx": 25, "volume_ratio": 1.6} 
    
    sentiment = news_filter.get_news_sentiment(active_sym)
    narrative = llm_narrative.generate_market_narrative(active_sym, dummy_ind, sentiment)
    
    st.info(narrative)
    st.markdown(f"<div style='font-size: 1.0rem;color: #000000'>Sentimento consolidado ({active_sym}): {sentiment:+.2f}</div>", unsafe_allow_html=True)

with ia_col2:
    st.markdown('<div class="section-header">💎 Elite Symbols Checklist</div>', unsafe_allow_html=True)
    elite_list = getattr(config, 'ELITE_SYMBOLS', [])
    if elite_list:
        # Se for dicionário, extrai as chaves
        symbols_to_show = list(elite_list.keys()) if isinstance(elite_list, dict) else elite_list
        st.write(", ".join(symbols_to_show[:8]) + "...")
        # Barra de progresso de cobertura
        coverage = min(100, (len(positions_df) / len(elite_list)) * 100) if elite_list else 0
        st.progress(coverage / 100, text=f"Exposição no Elite: {coverage:.1f}%")
    else:
        st.warning("⚠️ Lista ELITE_SYMBOLS não encontrada no config.")

# ===========================
# SEÇÃO 2: MÉTRICAS AVANÇADAS DE PERFORMANCE
# ===========================
st.markdown("---")
st.markdown('<div class="section-header">📈 Métricas Avançadas de Performance</div>', unsafe_allow_html=True)

metrics = calculate_advanced_metrics(historical_trades)
if metrics:
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric("🎯 Taxa Acerto", f"{metrics['win_rate']:.1f}%")
    
    with col2:
        st.metric("💹 Profit Factor", f"{metrics['profit_factor']:.22f}")
    
    with col3:
        st.metric("📊 Sharpe", f"{metrics['sharpe_ratio']:.2f}")
    
    with col4:
        st.metric("🍷 Sortino", f"{metrics['sortino_ratio']:.2f}",
                 help="Retorno ajustado pelo risco descendente")
    
    with col5:
        st.metric("🏆 Calmar", f"{metrics['calmar_ratio']:.2f}",
                 help="Retorno Anual / Max Drawdown")
    
    with col6:
        st.metric("📉 Max DD", f"R$ {metrics['max_drawdown']:.2f}")
    
    # Linha adicional de métricas
    st.markdown("### 📊 Estatísticas Detalhadas")
    
    col7, col8, col9, col10, col11, col12 = st.columns(6)
    
    with col7:
        st.metric("🟢 Ganho Médio", f"R$ {metrics['avg_win']:.2f}")
    
    with col8:
        st.metric("🔴 Perda Média", f"R$ {metrics['avg_loss']:.2f}")
    
    with col9:
        rr_ratio = metrics['avg_win'] / metrics['avg_loss'] if metrics['avg_loss'] > 0 else 0
        st.metric("⚖️ Risk/Reward", f"{rr_ratio:.2f}",
                 help="Relação ganho médio / perda média")
    
    with col10:
        st.metric("🏆 Melhor Trade", f"R$ {metrics['best_trade']:.2f}")
    
    with col11:
        st.metric("💔 Pior Trade", f"R$ {metrics['worst_trade']:.2f}")
    
    with col12:
        recovery_factor = abs(metrics['total_pnl'] / metrics['max_drawdown']) if metrics['max_drawdown'] != 0 else 0
        st.metric("🔄 Recovery Factor", f"{recovery_factor:.2f}",
                 help="Capacidade de recuperação")
else:
    st.info("📊 Aguardando histórico de operações para calcular métricas...")

# ===========================
# SEÇÃO 3: EQUITY CURVE E ANÁLISE DE DRAWDOWN
# ===========================
st.markdown("---")
st.markdown('<div class="section-header">📈 Curva de Equity e Análise de Drawdown</div>', unsafe_allow_html=True)

if not historical_trades.empty and 'P&L' in historical_trades.columns:
    # Filtrar apenas trades fechados
    closed_trades = historical_trades[historical_trades['Tipo'] == 'SAÍDA'].copy()
    
    if not closed_trades.empty:
        closed_trades = closed_trades.sort_values('Data')
        closed_trades['P&L Acumulado'] = closed_trades['P&L'].cumsum()
        closed_trades['Máximo Acumulado'] = closed_trades['P&L Acumulado'].cummax()
        closed_trades['Drawdown'] = closed_trades['P&L Acumulado'] - closed_trades['Máximo Acumulado']
        
        # Criar subplot com 2 gráficos
        fig_equity = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Curva de Equity', 'Underwater Chart (Drawdown)'),
            vertical_spacing=0.12,
            row_heights=[0.6, 0.4]
        )
        
        # Equity Curve
        fig_equity.add_trace(
            go.Scatter(
                x=list(range(len(closed_trades))),
                y=closed_trades['P&L Acumulado'],
                mode='lines',
                name='P&L Acumulado',
                line=dict(color='#667eea', width=3),
                fill='tozeroy',
                fillcolor='rgba(102, 126, 234, 0.1)'
            ),
            row=1, col=1
        )
        
        # Linha de máximos
        fig_equity.add_trace(
            go.Scatter(
                x=list(range(len(closed_trades))),
                y=closed_trades['Máximo Acumulado'],
                mode='lines',
                name='Máximo Histórico',
                line=dict(color='#00b894', width=2, dash='dash')
            ),
            row=1, col=1
        )
        
        # Underwater Chart
        fig_equity.add_trace(
            go.Scatter(
                x=list(range(len(closed_trades))),
                y=closed_trades['Drawdown'],
                mode='lines',
                name='Drawdown',
                line=dict(color='#d63031', width=2),
                fill='tozeroy',
                fillcolor='rgba(214, 48, 49, 0.2)'
            ),
            row=2, col=1
        )
        
        fig_equity.update_xaxes(title_text="Número da Operação", row=2, col=1)
        fig_equity.update_yaxes(title_text="P&L (R$)", row=1, col=1)
        fig_equity.update_yaxes(title_text="Drawdown (R$)", row=2, col=1)
        
        fig_equity.update_layout(
            height=700,
            showlegend=True,
            hovermode='x unified',
            template='plotly_white'
        )
        
        st.plotly_chart(fig_equity, width='stretch')
    else:
        st.info("📊 Aguardando trades fechados para análise de equity...")
else:
    st.info("📊 Aguardando histórico de trades...")

# ===========================
# SEÇÃO 4: ANÁLISE DE REJEIÇÕES (NOVO)
# ===========================
st.markdown("---")
st.markdown('<div class="section-header">🔍 Análise de Oportunidades Rejeitadas</div>', unsafe_allow_html=True)

@st.cache_data(ttl=60)
def load_rejections_data():
    """Lê o log de análises do dia e estrutura os dados"""
    today = datetime.now().strftime("%Y-%m-%d")
    log_file = Path(f"analysis_logs/analysis_log_{today}.txt")
    
    if not log_file.exists():
        return pd.DataFrame(), {}
    
    data = []
    reasons_count = {}
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Divide por blocos de entrada (separados por linhas de =)
        blocks = content.split('='*80)
        
        for block in blocks:
            if "|" not in block or "Motivo:" not in block:
                continue
                
            lines = [l.strip() for l in block.split('\n') if l.strip()]
            
            entry = {}
            for line in lines:
                if "|" in line and ("🕐" in line or "1" in line): # Timestamp line
                    parts = line.split("|")
                    if len(parts) >= 3:
                        entry['Time'] = parts[0].replace('🕐', '').strip()
                        entry['Symbol'] = parts[1].strip()
                        entry['Status'] = parts[2].strip()
                
                if "Sinal:" in line:
                    parts = line.split("|")
                    entry['Signal'] = parts[0].split(":")[1].strip()
                
                if "Filtros de Setup:" in line:
                     # Extract percentage if possible
                     try:
                         entry['Score'] = line.split("]")[1].split("%")[0].strip()
                     except:
                         entry['Score'] = "N/A"
                         
                if "Motivo:" in line:
                    reason = line.split("Motivo:")[1].strip()
                    entry['Reason'] = reason
                    reasons_count[reason] = reasons_count.get(reason, 0) + 1

                if "RSI:" in line:
                    entry['RSI'] = line.split("RSI:")[1].strip()
                
                if "Volume:" in line:
                    entry['Volume'] = line.split("Volume:")[1].strip()
            
            if entry and 'Status' in entry and ('REJEITADA' in entry['Status'] or 'AGUARDANDO' in entry['Status']):
                data.append(entry)
                
    except Exception as e:
        logger.error(f"Erro ao ler logs: {e}")
        
    return pd.DataFrame(data), reasons_count

rejections_df, reasons_stats = load_rejections_data()

if not rejections_df.empty:
    r_col1, r_col2 = st.columns([2, 1])
    
    with r_col1:
        st.markdown("### 📋 Últimas Rejeições")
        st.dataframe(
            rejections_df[['Time', 'Symbol', 'Signal', 'Score', 'Reason', 'RSI', 'Volume']],
            width='stretch',
            height=300
        )
    
    with r_col2:
        st.markdown("### 📉 Principais Motivos")
        if reasons_stats:
            reasons_df = pd.DataFrame(
                list(reasons_stats.items()), 
                columns=['Motivo', 'Contagem']
            ).sort_values('Contagem', ascending=False)
            
            fig_reasons = px.pie(
                reasons_df, 
                values='Contagem', 
                names='Motivo',
                hole=0.4,
                color_discrete_sequence=px.colors.sequential.RdBu
            )
            fig_reasons.update_layout(
                margin=dict(t=0, b=0, l=0, r=0),
                height=250,
                showlegend=False
            )
            st.plotly_chart(fig_reasons, width='stretch')
            
            # Top 3 reasons text
            st.markdown("#### Top Bloqueios:")
            for i, row in reasons_df.head(3).iterrows():
                st.markdown(f"- **{row['Contagem']}x**: {row['Motivo']}")
else:
    st.info("✅ Nenhuma rejeição registrada hoje (ou arquivo de log ainda vazio).")

# ===========================
# SEÇÃO 5: ANÁLISE POR ATIVO
# ===========================
st.markdown("---")
st.markdown('<div class="section-header">🎯 Performance por Ativo</div>', unsafe_allow_html=True)

if not historical_trades.empty and 'Símbolo' in historical_trades.columns:
    # Análise por ativo
    symbol_analysis = historical_trades[historical_trades['Tipo'] == 'SAÍDA'].groupby('Símbolo').agg({
        'P&L': ['sum', 'mean', 'count'],
        'Lado': lambda x: (x == 'COMPRA').sum() / len(x) * 100 if len(x) > 0 else 0
    }).round(2)
    
    
    col_sym1, col_sym2 = st.columns([2, 1])
    
    with col_sym1:
        # Gráfico de barras horizontal
        fig_symbols = go.Figure()
        
        colors = ['#00b894' if x > 0 else '#d63031' for x in symbol_analysis['P&L Total']]
        
        fig_symbols.add_trace(go.Bar(
            y=symbol_analysis.index,
            x=symbol_analysis['P&L Total'],
            orientation='h',
            marker=dict(color=colors),
            text=symbol_analysis['P&L Total'].apply(lambda x: f'R$ {x:,.2f}'),
            textposition='outside'
        ))
        
        fig_symbols.update_layout(
            title='P&L Total por Ativo',
            xaxis_title='P&L (R$)',
            yaxis_title='Ativo',
            height=400,
            showlegend=False,
            template='plotly_white'
        )
        
        st.plotly_chart(fig_symbols, width='stretch')
    
    with col_sym2:
        st.markdown("### 📊 Top Ativos")
        
        # Formatar tabela
        styled_symbols = symbol_analysis.style.format({
            'P&L Total': 'R$ {:,.2f}',
            'P&L Médio': 'R$ {:,.2f}',
            'Win Rate': '{:.1f}%',
            '% Compras': '{:.1f}%'
        })
        
        st.dataframe(styled_symbols, width='stretch')
else:
    st.info("📊 Aguardando histórico para análise por ativo...")

# ===========================
# SEÇÃO 5: HEATMAP E ANÁLISE TEMPORAL
# ===========================
st.markdown("---")
st.markdown('<div class="section-header">🔥 Heatmap & Análise Temporal</div>', unsafe_allow_html=True)

if not historical_trades.empty and 'Hora' in historical_trades.columns:
    try:
        # Prepara dados
        df_hm = historical_trades[historical_trades['Tipo'] == 'SAÍDA'].copy()
        
        # Extrai hora apenas
        if not df_hm.empty:
            df_hm['Hour'] = pd.to_datetime(df_hm['Hora'], format='%H:%M:%S').dt.hour
            
            # Agrupamento para Heatmap (Símbolo x Hora -> Win Rate)
            heatmap_data = df_hm.groupby(['Símbolo', 'Hour']).agg({
                'P&L': lambda x: (x > 0).sum() / len(x) * 100 # Win Rate
            }).reset_index()
            heatmap_data.columns = ['Símbolo', 'Hour', 'Win Rate']
            
            # Pivot para matriz do heatmap
            heatmap_matrix = heatmap_data.pivot(index='Símbolo', columns='Hour', values='Win Rate')
            
            # Layout: Heatmap + Gráfico Horário original
            col_hm1, col_hm2 = st.columns([2, 1])
            
            with col_hm1:
                st.markdown("### 🔥 Win Rate por Símbolo x Horário")
                if not heatmap_matrix.empty:
                    fig_hm = px.imshow(
                        heatmap_matrix,
                        labels=dict(x="Hora do Dia", y="Ativo", color="Win Rate (%)"),
                        x=heatmap_matrix.columns,
                        y=heatmap_matrix.index,
                        color_continuous_scale="RdYlGn",
                        text_auto=".0f",
                        aspect="auto"
                    )
                    fig_hm.update_layout(height=400)
                    st.plotly_chart(fig_hm, width='stretch')
                else:
                    st.info("Dados insuficientes para heatmap.")
                
            with col_hm2:
                st.markdown("### ⏰ Lucro por Hora")
                hourly_stats = df_hm.groupby('Hour')['P&L'].sum().reset_index()
                if not hourly_stats.empty:
                    fig_bar = px.bar(
                        hourly_stats, x='Hour', y='P&L',
                        color='P&L',
                        color_continuous_scale=["red", "green"],
                        labels={'P&L': 'Lucro Total (R$)'}
                    )
                    fig_bar.update_layout(height=400)
                    st.plotly_chart(fig_bar, width='stretch')
                else:
                    st.info("Dados insuficientes para gráfico horário.")

    except Exception as e:
        st.error(f"Erro ao gerar Heatmap: {e}")
else:
    st.info("📊 Aguardando histórico para análise temporal...")

# ===========================
# SEÇÃO 6: SIMULADOR MONTE CARLO (PREDICTIVE WR)
# ===========================
st.markdown("---")
st.markdown('<div class="section-header">🔮 Simulador de Cenários Futuros (Monte Carlo)</div>', unsafe_allow_html=True)

if metrics:
    c1, c2, c3 = st.columns(3)
    with c1:
        sim_balance = st.number_input("Capital Inicial", value=float(acc['equity']), step=1000.0)
    with c2:
        sim_trades = st.slider("Número de Trades Futuros", 20, 500, 100)
    with c3:
        sim_risk_pct = st.slider("Risco por Trade (%)", 0.5, 5.0, 1.0) / 100

    if st.button("🚀 Executar Simulação Preditiva", width='stretch'):
        try:
            # Parâmetros baseados no histórico
            hist_win_rate = metrics['win_rate'] / 100
            avg_win = metrics['avg_win']
            avg_loss = abs(metrics['avg_loss'])
            
            # Se histórico insuficiente, usa parâmetros padrão conservadores
            if hist_win_rate == 0:
                hist_win_rate = 0.5
                avg_win = 100
                avg_loss = 100
                st.warning("Histórico insuficiente. Usando parâmetros padrão (WR 50%, RR 1:1)")

            # Simulação Monte Carlo (100 cenários)
            scenarios = 100
            simulation_results = []
            
            for _ in range(scenarios):
                equity_curve = [sim_balance]
                current_equity = sim_balance
                
                # Gera sequência aleatória de resultados
                results = np.random.random(sim_trades)
                
                for r in results:
                    risk_amount = current_equity * sim_risk_pct
                    
                    if r < hist_win_rate:
                        # Win
                        rr_ratio = avg_win / avg_loss if avg_loss > 0 else 1.0
                        profit = risk_amount * rr_ratio 
                        profit *= np.random.normal(1.0, 0.2)
                        current_equity += profit
                    else:
                        # Loss
                        loss = risk_amount
                        loss *= np.random.normal(1.0, 0.1)
                        current_equity -= loss
                    
                    equity_curve.append(current_equity)
                
                simulation_results.append(equity_curve)
            
            # Plot
            fig_sim = go.Figure()
            
            # Adiciona trilhas
            for sim in simulation_results:
                fig_sim.add_trace(go.Scatter(
                    y=sim, 
                    mode='lines', 
                    line=dict(color='rgba(100, 100, 255, 0.1)', width=1),
                    showlegend=False
                ))
            
            # Média
            avg_curve = np.mean(simulation_results, axis=0)
            fig_sim.add_trace(go.Scatter(
                y=avg_curve,
                mode='lines',
                name='Cenário Médio',
                line=dict(color='#0f3460', width=3)
            ))
            
            fig_sim.update_layout(
                title=f"Projeção para próximos {sim_trades} trades (Baseado em WR {hist_win_rate:.1%})",
                xaxis_title="Número de Trades",
                yaxis_title="Equity (R$)",
                template="plotly_white",
                height=500
            )
            
            st.plotly_chart(fig_sim, width='stretch')
            
            # Estatísticas
            final_values = [s[-1] for s in simulation_results]
            p90 = np.percentile(final_values, 90)
            p10 = np.percentile(final_values, 10)
            prob_profit = sum(1 for v in final_values if v > sim_balance) / scenarios * 100
            
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("Média Final", f"R$ {np.mean(final_values):,.2f}", delta=f"{(np.mean(final_values)/sim_balance - 1)*100:.1f}%")
            sc2.metric("Pior Caso (10%)", f"R$ {p10:,.2f}")
            sc3.metric("Probabilidade Lucro", f"{prob_profit:.1f}%")

        except Exception as e:
            st.error(f"Erro na simulação: {e}")

# ===========================
# SEÇÃO 6: POSIÇÕES ABERTAS DETALHADAS
# ===========================
st.markdown("---")
st.markdown('<div class="section-header">📋 Posições Abertas - Gestão Ativa</div>', unsafe_allow_html=True)

if not positions_df.empty:
    # Adicionar análise de risco por posição
    positions_display = positions_df.copy()
    
    # Calcular exposição percentual
    total_exposure = (positions_display['Volume'] * positions_display['Preço Atual']).sum()
    positions_display['Exposição %'] = (positions_display['Volume'] * positions_display['Preço Atual']) / total_exposure * 100
    
    # Status da posição
    def get_position_status(row):
        if row['P&L %'] > 2:
            return '🟢 Em Lucro Forte'
        elif row['P&L %'] > 0:
            return '🟡 Em Lucro'
        elif row['P&L %'] > -1:
            return '🟠 Pequena Perda'
        else:
            return '🔴 Em Perda'
    
    positions_display['Status'] = positions_display.apply(get_position_status, axis=1)

    def get_close_gap_status(row):
        now = datetime.now()
        close_all_by = getattr(config, 'FRIDAY_CLOSE_ALL_BY', getattr(config, 'CLOSE_ALL_BY', None)) if now.weekday() == 4 else getattr(config, 'CLOSE_ALL_BY', None)
        max_candles = int(getattr(config, 'MAX_TRADE_DURATION_CANDLES', 0) or 0)
        
        opened = datetime.fromtimestamp(row.get('Timestamp', now.timestamp()))
        time_open_minutes = (now - opened).total_seconds() / 60.0
        candles_open = int(time_open_minutes / 15)

        close_trigger = None
        minutes_to_close = None
        
        if close_all_by:
            try:
                close_by_time = datetime.strptime(close_all_by, "%H:%M").time()
                close_by_dt = datetime.combine(now.date(), close_by_time)
                minutes_to_close = max(0.0, (close_by_dt - now).total_seconds() / 60.0)
                if now.time() >= close_by_time:
                    close_trigger = f"Day Close ({close_all_by})"
            except Exception:
                minutes_to_close = None

        if max_candles > 0 and candles_open >= max_candles:
            close_trigger = f"Time-stop ({candles_open}/{max_candles} candles)"
        
        if close_trigger:
            return f"TRIGGER: {close_trigger}"
        
        if max_candles > 0:
            candles_remaining = max(0, max_candles - candles_open)
            time_stop_txt = f"faltam {candles_remaining}c p/ time-stop"
        else:
            time_stop_txt = "time-stop N/A"
        
        day_close_txt = f"faltam {minutes_to_close:.0f}m p/ day close" if minutes_to_close is not None else "day close N/A"
        return f"{time_stop_txt}; {day_close_txt}"

    positions_display['Fechamento'] = positions_display.apply(get_close_gap_status, axis=1)
    
    # Formatação de tabela
    cols_to_show = ['Símbolo', 'Lado', 'Volume', 'Preço Entrada', 'Preço Atual',
                    'Stop Loss', 'P&L R$', 'P&L %', 'Exposição %', 'Tempo Aberto', 'Fechamento', 'Status']
    
    positions_display = positions_display[cols_to_show]
    
    styled_positions = positions_display.style.format({
        'Volume': '{:.2f}',
        'Preço Entrada': '{:.5f}',
        'Preço Atual': '{:.5f}',
        'Stop Loss': '{:.5f}',
        'P&L R$': 'R$ {:+,.2f}',
        'P&L %': '{:+.2f}%',
        'Exposição %': '{:.1f}%'
    })
    
    st.dataframe(styled_positions, width='stretch', hide_index=True)
    
    # Gráfico de exposição
    col_exp1, col_exp2 = st.columns(2)
    
    with col_exp1:
        fig_exposure = px.pie(
            positions_display,
            values='Exposição %',
            names='Símbolo',
            title='Distribuição de Exposição por Ativo'
        )
        fig_exposure.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_exposure, width='stretch')
    
    with col_exp2:
        # P&L por símbolo em posições abertas
        fig_pnl_open = px.bar(
            positions_display,
            x='Símbolo',
            y='P&L R$',
            color='P&L R$',
            color_continuous_scale=['#d63031', '#00b894'],
            title='P&L Flutuante por Ativo'
        )
        st.plotly_chart(fig_pnl_open, width='stretch')
    
    # Exportação
    csv_positions = positions_display.to_csv(index=False).encode('utf-8')
    clicked_positions = st.download_button(
        label="📥 Exportar Posições (CSV)",
        data=csv_positions,
        file_name=f"posicoes_abertas_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
        key="download_positions_csv"
    )
    try:
        b64 = base64.b64encode(csv_positions).decode()
        href = f'data:text/csv;base64,{b64}'
        st.markdown(f'<a href="{href}" download="posicoes_abertas_{datetime.now().strftime("%Y%m%d_%H%M")}.csv">🔗 Download CSV (compatível)</a>', unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Falha ao preparar link alternativo de download: {e}")
else:
    st.info("✅ Nenhuma posição aberta no momento. Mercado em observação.")

# ===========================
# SEÇÃO 7: ANÁLISE PSICOLÓGICA
# ===========================
st.markdown("---")
st.markdown('<div class="section-header">🧠 Análise Psicológica - Disciplina e Controle Emocional</div>', unsafe_allow_html=True)

if metrics and metrics.get('total_trades', 0) > 5:
    col_psych1, col_psych2, col_psych3 = st.columns(3)
    
    with col_psych1:
        st.markdown("### 🔥 Sequências")
        st.metric("Maior sequência de vitórias", f"{metrics['win_streak']} trades")
        st.metric("Maior sequência de perdas", f"{metrics['loss_streak']} trades")
        
        # Alerta de revenge trading
        if metrics['loss_streak'] > 3:
            st.warning("⚠️ **Alerta:** Sequência de perdas elevada. Considere pausar e revisar estratégia.")
    
    with col_psych2:
        st.markdown("### 📊 Consistência")
        
        # Calcular desvio padrão dos resultados
        if not historical_trades.empty:
            closed = historical_trades[historical_trades['Tipo'] == 'SAÍDA']
            if len(closed) > 1:
                std_dev = closed['P&L'].std()
                consistency_score = 100 - min((std_dev / abs(closed['P&L'].mean()) * 100), 100) if closed['P&L'].mean() != 0 else 0
                
                st.metric("Score de Consistência", f"{consistency_score:.1f}/100")
                
                if consistency_score > 70:
                    st.success("✅ Trading consistente")
                elif consistency_score > 50:
                    st.info("ℹ️ Consistência moderada")
                else:
                    st.warning("⚠️ Alta volatilidade nos resultados")
    
    with col_psych3:
        st.markdown("### 🎯 Gestão de Risco")
        
        # Verificar se está respeitando limites
        risk_per_trade = getattr(config, 'RISK_PER_TRADE_PCT', 0.02)
        max_dd = getattr(config, 'MAX_DAILY_DRAWDOWN_PCT', 0.05)
        
        if BOT_CONNECTED and daily_max_equity > 0:
            current_dd = (acc['equity'] - daily_max_equity) / daily_max_equity
            dd_usage = abs(current_dd / max_dd * 100)
            
            st.metric("Uso do Drawdown Diário", f"{dd_usage:.1f}%")
            
            if dd_usage > 80:
                st.error("🚨 Próximo do limite de drawdown!")
            elif dd_usage > 50:
                st.warning("⚠️ Drawdown em nível de atenção")
            else:
                st.success("✅ Drawdown sob controle")
        
        # Verificar tamanho das posições
        if not positions_df.empty:
            avg_position_size = positions_df['Volume'].mean()
            st.metric("Tamanho médio de posição", f"{avg_position_size:.2f}")
else:
    st.info("🧠 Execute mais operações para análise psicológica detalhada (mínimo: 5 trades)")

# ===========================
# SEÇÃO 8: COMPARAÇÃO COM BENCHMARKS
# ===========================
st.markdown("---")
st.markdown('<div class="section-header">🏆 Comparação com Benchmarks</div>', unsafe_allow_html=True)

col_bench1, col_bench2, col_bench3, col_bench4 = st.columns(4)
# Benchmarks simulados (em produção, buscar dados reais via API ou MT5)
ibov_return = 0.15 # Exemplo
spy_return = 0.08
cdi_return = 0.03
with col_bench1:
    st.markdown("### 📊 Sua Performance")
    if metrics:
        period_return = (metrics['total_pnl'] / acc['balance'] * 100) if acc['balance'] > 0 else 0
        st.metric("Retorno no Período", f"{period_return:.2f}%")
with col_bench2:
    st.markdown("### 🇧🇷 IBOVESPA")
    st.metric("Retorno", f"{ibov_return:.2f}%")
    if metrics:
        outperformance = period_return - ibov_return
        st.metric("vs. Sua Performance", f"{outperformance:+.2f}%")
with col_bench3:
    st.markdown("### 🇺🇸 S&P 500")
    st.metric("Retorno", f"{spy_return:.2f}%")
    if metrics:
        outperformance = period_return - spy_return
        st.metric("vs. Sua Performance", f"{outperformance:+.2f}%")
with col_bench4:
    st.markdown("### 🇧🇷 CDI")
    st.metric("Retorno", f"{cdi_return:.2f}%")
    if metrics:
        outperformance = period_return - cdi_return
        st.metric("vs. Sua Performance", f"{outperformance:+.2f}%")

# ===========================
# SEÇÃO 9: DADOS DE MERCADO EM TEMPO REAL
# ===========================
st.markdown("---")
st.markdown('<div class="section-header">📈 Dados de Mercado em Tempo Real</div>', unsafe_allow_html=True)

# Carregar dados reais
real_time_df = load_real_time_market_data()

if not real_time_df.empty:
    # Formatação
    styled_real_time = real_time_df.style.format({
        'Bid': '{:.2f}',
        'Ask': '{:.2f}',
        'Último': '{:.2f}',
        'Volume': '{:,.0f}'
    })
    
    st.dataframe(styled_real_time, width='stretch', hide_index=True)
    
    # Gráfico de preços
    fig_real_time = px.bar(
        real_time_df,
        x='Símbolo',
        y='Último',
        title='Preços em Tempo Real',
        labels={'Último': 'Preço Último (R$)'}
    )
    st.plotly_chart(fig_real_time, width='stretch')
else:
    st.info("📊 Aguardando dados de mercado em tempo real... Verifique conexão MT5.")

# ===========================
# SEÇÃO 10: JOURNAL DE TRADING
# ===========================
st.markdown("---")
st.markdown('<div class="section-header">📓 Journal de Trading - Registre suas Observações</div>', unsafe_allow_html=True)

journal_file = "trading_journal.json"
if os.path.exists(journal_file):
    with open(journal_file, 'r') as f:
        journal = json.load(f)
else:
    journal = []

# Formulário de nova entrada
with st.form(key='journal_form'):
    observation = st.text_area("Observação sobre o dia de trading")
    emotion = st.selectbox("Estado emocional", ["Confiante", "Ansioso", "Frustrado", "Neutro", "Outro"])
    lessons = st.text_area("Lições aprendidas")
    submit = st.form_submit_button("📝 Adicionar ao Journal")
    
    if submit:
        new_entry = {
            'data': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'observacao': observation,
            'emocao': emotion,
            'licoes': lessons
        }
        journal.append(new_entry)
        with open(journal_file, 'w') as f:
            json.dump(journal, f, indent=4)
        st.success("✅ Entrada adicionada ao journal!")

# Mostrar journal
if journal:
    journal_df = pd.DataFrame(journal)
    journal_df = journal_df.sort_values('data', ascending=False)
    st.dataframe(
        journal_df,
        width='stretch',
        hide_index=True,
        column_config={
            'data': "Data",
            'observacao': "Observação",
            'emocao': "Emoção",
            'licoes': "Lições"
        }
    )
else:
    st.info("📓 Seu journal está vazio. Comece registrando suas observações!")

# ===========================
# SEÇÃO 11: RELATÓRIOS EXPORTÁVEIS
# ===========================
st.markdown("---")
st.markdown('<div class="section-header">📑 Relatórios Exportáveis</div>', unsafe_allow_html=True)

col_rep1, col_rep2 = st.columns(2)

with col_rep1:
    if st.button("📄 Gerar Relatório PDF Completo"):
        st.info("Em desenvolvimento: Relatório com todas as métricas e gráficos.")

with col_rep2:
    if st.button("📊 Exportar Dados para Excel"):
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            historical_trades.to_excel(writer, sheet_name='Histórico Trades', index=False)
            positions_df.to_excel(writer, sheet_name='Posições Abertas', index=False)
            pd.DataFrame([metrics]).to_excel(writer, sheet_name='Métricas', index=False)
            if 'journal_df' in locals():
                journal_df.to_excel(writer, sheet_name='Journal', index=False)
        excel_data = output.getvalue()
        clicked_excel = st.download_button(
            label="📥 Baixar Excel",
            data=excel_data,
            file_name=f"trading_report_{datetime.now().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download_excel_report"
        )
        try:
            b64 = base64.b64encode(excel_data).decode()
            href = f'data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}'
            st.markdown(f'<a href="{href}" download="trading_report_{datetime.now().strftime("%Y%m%d")}.xlsx">🔗 Download Excel (compatível)</a>', unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"Falha ao preparar link alternativo de download: {e}")
# ===========================
# SEÇÃO: DIAGNÓSTICO DE OPORTUNIDADES (Por que não comprou?)
# ===========================
# ===========================
# FUNÇÃO AUXILIAR DE DIAGNÓSTICO
# ===========================
def diagnosticar_ativo(symbol):
    """
    Realiza um diagnóstico rápido do ativo para explicar por que o bot pode não estar entrando.
    Retorna uma lista de dicionários com os checks.
    """
    checks = []
    
    # 1. Verificação de Conexão e Simbolo
    # Garantir conexão
    if not mt5.terminal_info():
        mt5.initialize(path=config.MT5_TERMINAL_PATH)
        
    found = mt5.symbol_select(symbol, True)
    
    # Fallback para fracionário (Ex: ITUB4 -> ITUB4F) se não achar padrão
    if not found and not symbol.endswith('F'):
        symbol_frac = symbol + 'F'
        if mt5.symbol_select(symbol_frac, True):
            symbol = symbol_frac
            found = True
            
    if not found:
        checks.append({
            'Regra': "Market Watch", 'Valor': "N/A", 'Meta': "Presente", 
            'Ok': False, 'Obs': f"Ativo '{symbol}' não encontrado"
        })
        return checks
    
    checks.append({
        'Regra': "Market Watch", 'Valor': "OK", 'Meta': "Presente", 
        'Ok': True, 'Obs': f"Disponível ({symbol})"
    })

    # 2. Dados de Preço (M15)
    # Tenta pegar dados via utils ou direto do mt5 se falhar
    try:
        df = utils.safe_copy_rates(symbol, mt5.TIMEFRAME_M15, 100)
    except Exception:
        df = None

    if not utils.is_valid_dataframe(df):
        checks.append({
            'Regra': "Dados M15", 'Valor': "Vazio", 'Meta': "100 candles", 
            'Ok': False, 'Obs': "Falha ao baixar dados"
        })
        return checks

    try:
        # Preparar indicadores básicos
        close = df['close']
        
        # Parâmetros do ativo (ou default)
        params = config.ELITE_SYMBOLS.get(symbol, {})
        # Fallback para WIN/WDO generics se necessario
        if not params:
            if "WIN" in symbol: params = config.ELITE_SYMBOLS.get("WIN$N", {})
            elif "WDO" in symbol: params = config.ELITE_SYMBOLS.get("WDO$N", {})
        
        # Defaults
        ema_s_period = params.get('ema_short', 12)
        ema_l_period = params.get('ema_long', 72)
        rsi_period = 14
        min_vol = getattr(config, 'MIN_AVG_VOLUME_20', 300000)

        # Cálculos EMA
        ema_short = close.ewm(span=ema_s_period).mean().iloc[-1]
        ema_long = close.ewm(span=ema_l_period).mean().iloc[-1]
        
        # Cálculos RSI (Manual para garantir)
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / loss
        rsi_val = 100 - (100 / (1 + rs)).iloc[-1]
        if pd.isna(rsi_val): rsi_val = 50.0
        
        # Volume (CORRIGIDO: Volume Financeiro, não tick_volume)
        # MIN_AVG_VOLUME_20 = 300k significa R$ 300k, não 300k contratos
        vol_financial = (df['tick_volume'] * df['close']).rolling(20).mean().iloc[-1]
        if pd.isna(vol_financial): vol_financial = 0
        
        # 3. Tendência (EMA)
        trend_state = "Alta" if ema_short > ema_long else "Baixa"
        dist_ema = abs(ema_short - ema_long) / ema_long * 100
        checks.append({
            'Regra': f"Tendência (EMA {ema_s_period}/{ema_l_period})", 
            'Valor': trend_state, 
            'Meta': "Definida", 
            'Ok': True, 
            'Obs': f"Dist: {dist_ema:.2f}%"
        })

        # 4. Filtro de Volume (Financeiro)
        checks.append({
            'Regra': "Liquidez (Vol 20 R$)", 
            'Valor': f"R$ {vol_financial:,.0f}", 
            'Meta': f">R$ {min_vol:,.0f}", 
            'Ok': vol_financial > min_vol, 
            'Obs': "Volume Baixo" if vol_financial <= min_vol else "Ok"
        })
        
        # 5. RSI Range
        rsi_low = params.get('rsi_low', 30)
        rsi_high = params.get('rsi_high', 70)
        
        rsi_msg = "Neutro"
        if rsi_val < rsi_low: rsi_msg = "Sobrevendido (Pode comprar)"
        elif rsi_val > rsi_high: rsi_msg = "Sobrecomprado (Pode vender)"
        
        checks.append({
            'Regra': "RSI (14)", 
            'Valor': f"{rsi_val:.1f}", 
            'Meta': f"{rsi_low}-{rsi_high}", 
            'Ok': True, # RSI nunca bloqueia "totalmente", só define lado
            'Obs': rsi_msg
        })
        
    except Exception as e:
        checks.append({
            'Regra': "Cálculo Indicadores", 
            'Valor': "Erro", 
            'Meta': "Sucesso", 
            'Ok': False, 
            'Obs': str(e)[0:20]
        })
    
    return checks

st.markdown("---")
st.markdown('<div class="section-header">🔍 Raio-X de Execução (Log Diário - Why no trade?)</div>', unsafe_allow_html=True)

with st.expander("📂 Abrir Detalhes de Análise de Sinais (Logs Reais do Dia)", expanded=True):
    df_logs = load_execution_logs()
    
    if not df_logs.empty:
        c1, c2 = st.columns([1, 1])
        with c1:
            all_symbols = ["Todos"] + list(df_logs['Symbol'].unique())
            sel_symbol = st.selectbox("🎯 Filtrar Ativo", all_symbols, key="sel_sym_log")
        with c2:
            all_statuses = ["Todos"] + list(df_logs['Status'].unique())
            sel_status = st.selectbox("🚦 Filtrar Status", all_statuses, key="sel_stat_log")
            
        # Filtragem
        filtered = df_logs.copy()
        if sel_symbol != "Todos": filtered = filtered[filtered['Symbol'] == sel_symbol]
        if sel_status != "Todos": filtered = filtered[filtered['Status'] == sel_status]
        
        # Display Principal
        st.dataframe(
            filtered[['Time', 'Symbol', 'Status', 'Reason', 'Score']], 
            width="stretch",
            hide_index=True,
            column_config={
                "Time": "Hora",
                "Reason": "Motivo / Detalhe Real do Log",
                "Score": st.column_config.ProgressColumn(
                    "Setup Score", format="%d", min_value=0, max_value=120
                )
            }
        )
        
        # Detalhes do Último Item Selecionado (ou primeiro da lista)
        if not filtered.empty:
            st.markdown("#### 🔬 Insights do Log selecionado")
            last = filtered.iloc[0]
            
            # Cards de Indicadores do Log
            k1, k2, k3 = st.columns(3)
            k1.metric("RSI (Log)", last.get('RSI', '-'))
            k2.metric("ADX (Log)", last.get('ADX', '-'))
            k3.warning(f"Decisão: {last['Status']}")
            
            st.info(f"💬 **Motivo Original:** {last['Reason']}")
            
            with st.status("📜 Visualizar Entrada Completa do Log", expanded=False):
                st.code(last['FullLog'], language='text')
    else:
        st.info("📭 Nenhum registro de análise encontrado hoje nos logs. O bot está gerando arquivos em `analysis_logs/`?")

st.markdown("---")
st.markdown('<div class="section-header">🔎 Raio X do Porquê Não Comprou</div>', unsafe_allow_html=True)
def _get_symbols_list():
    syms = []
    try:
        if BOT_CONNECTED and hasattr(config, "ELITE_SYMBOLS"):
            syms = list(getattr(config, "ELITE_SYMBOLS", {}).keys())
    except Exception:
        pass
    if not syms and not positions_df.empty and "Símbolo" in positions_df.columns:
        syms = sorted(list(set(positions_df["Símbolo"].tolist())))
    if not syms and hasattr(config, "SYMBOLS"):
        syms = list(getattr(config, "SYMBOLS", []))
    return syms
def _get_indicators(symbol):
    ind = {}
    try:
        if BOT_CONNECTED and "bot_state" in globals():
            try:
                ind = bot_state.get_indicators(symbol)
            except Exception:
                pass
        if not ind or ind.get("error"):
            try:
                df = utils.safe_copy_rates(symbol, mt5.TIMEFRAME_M15, 300)
                if df is None or len(df) <= 50:
                    try:
                        if not mt5.terminal_info():
                            mt5.initialize(path=getattr(config, "MT5_TERMINAL_PATH", None))
                        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 300) or []
                        df = pd.DataFrame(rates)
                    except Exception:
                        df = None
                if df is not None and len(df) > 50:
                    close = df["close"]
                    emaf = close.ewm(span=21, adjust=False).mean().iloc[-1]
                    emas = close.ewm(span=50, adjust=False).mean().iloc[-1]
                    rsi = None
                    try:
                        if hasattr(utils, "get_rsi"):
                            rsi = utils.get_rsi(df)
                        if rsi is None:
                            delta = close.diff()
                            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                            rs = gain / loss
                            rsi = float(100 - (100 / (1 + rs)).iloc[-1])
                    except Exception:
                        rsi = None
                    adx = None
                    try:
                        if hasattr(utils, "get_adx"):
                            adx = utils.get_adx(df)
                        if adx is None:
                            hi = df["high"]
                            lo = df["low"]
                            prev_close = close.shift(1)
                            tr = pd.concat([(hi - lo).abs(), (hi - prev_close).abs(), (lo - prev_close).abs()], axis=1).max(axis=1)
                            plus_dm = (hi.diff()).clip(lower=0)
                            minus_dm = (-lo.diff()).clip(lower=0)
                            roll = 14
                            atr = tr.rolling(roll).mean()
                            plus_di = 100 * (plus_dm.rolling(roll).mean() / atr)
                            minus_di = 100 * (minus_dm.rolling(roll).mean() / atr)
                            dx = (abs(plus_di - minus_di) / (plus_di + minus_di)).replace([np.inf, np.nan], 0) * 100
                            adx = float(dx.rolling(roll).mean().iloc[-1])
                    except Exception:
                        adx = None
                    vwap = None
                    try:
                        if hasattr(utils, "get_intraday_vwap"):
                            vwap = utils.get_intraday_vwap(df)
                        if vwap is None and {"tick_volume", "close"}.issubset(df.columns):
                            v = df["tick_volume"]
                            p = df["close"]
                            vwap = float((p * v).sum() / max(v.sum(), 1))
                    except Exception:
                        vwap = None
                    vol_ratio = None
                    try:
                        if "tick_volume" in df.columns:
                            vol20 = df["tick_volume"].rolling(20).mean().iloc[-1]
                            cur = float(df["tick_volume"].iloc[-1])
                            base = float(vol20) if vol20 and vol20 > 0 else 0
                            vol_ratio = float(cur / base) if base > 0 else 0
                    except Exception:
                        vol_ratio = None
                    ind = {"rsi": rsi, "adx": adx, "ema_fast": emaf, "ema_slow": emas, "vwap": vwap, "volume_ratio": vol_ratio}
            except Exception:
                pass
    except Exception:
        pass
    return ind or {}
def _fmt_val(v):
    try:
        if v is None:
            return "N/A"
        if isinstance(v, (int, float)):
            return f"{float(v):.2f}"
        return str(v)
    except Exception:
        return str(v)
def _badge_row(name, ok, value, threshold, tooltip):
    color = "#00b894" if ok else "#d63031"
    bg = f"{color}20"
    icon = "✅" if ok else "❌"
    html = f"""
    <div style="display:flex;justify-content:space-between;align-items:center;padding:8px;border-left:4px solid {color};background:{bg};border-radius:8px;margin-bottom:6px;">
        <div style="display:flex;align-items:center;gap:8px;">
            <span style="font-weight:600;color:#1a1a2e;">{icon} {name}</span>
            <span title="{tooltip}" style="color:#636e72;">ℹ︎</span>
        </div>
        <div style="display:flex;gap:12px;">
            <span style="font-family:monospace;color:#2d3436;">{_fmt_val(value)}</span>
            <span style="font-family:monospace;color:#2d3436;">{_fmt_val(threshold)}</span>
        </div>
    </div>
    """
    return html
def _compute_criteria(symbol, side):
    now = datetime.now()
    ind = _get_indicators(symbol)
    rsi = ind.get("rsi")
    adx = ind.get("adx")
    emaf = ind.get("ema_fast")
    emas = ind.get("ema_slow")
    ema_trend = "UP" if (emaf is not None and emas is not None and emaf > emas) else "DOWN"
    vwap = ind.get("vwap")
    vol_ratio = ind.get("volume_ratio")
    score = ind.get("score", 0)
    atr = ind.get("atr", None)
    hour = now.hour
    if hour < 12:
        min_vol = 1.2
        period = "Manhã"
    elif 12 <= hour < 14:
        min_vol = float(getattr(config, "LUNCH_MIN_VOLUME_RATIO", 0.5) or 0.5)
        period = "Almoço"
    else:
        min_vol = 0.8
        period = "Tarde"
    base_min_score = float(getattr(config, "MIN_SIGNAL_SCORE", 35) or 35)
    if period == "Manhã":
        min_score = base_min_score + float(getattr(config, "ENTRY_SCORE_DELTA_MORNING", 5) or 5)
    elif period == "Almoço":
        min_score = base_min_score + float(getattr(config, "ENTRY_SCORE_DELTA_LUNCH", 10) or 10)
    else:
        min_score = base_min_score + float(getattr(config, "ENTRY_SCORE_DELTA_AFTERNOON", 0) or 0)
    tech = []
    tech.append(("Tendência EMA", (side == "BUY" and ema_trend == "UP") or (side == "SELL" and ema_trend == "DOWN"), ema_trend, side, "Alinhamento da tendência entre EMAs"))
    if rsi is not None:
        tech.append(("RSI não esticado", (side == "BUY" and rsi <= 70) or (side == "SELL" and rsi >= 30), rsi, "BUY<=70 | SELL>=30", "Prevenção de exaustão de preço"))
    if adx is not None:
        tech.append(("ADX mínimo", adx >= 20, adx, ">=20", "Força de tendência suficiente"))
    if vwap is not None:
        price_ok = False
        try:
            t = mt5.symbol_info_tick(symbol)
            if t:
                px = t.bid if side == "SELL" else t.ask
                price_ok = (side == "BUY" and px > vwap) or (side == "SELL" and px < vwap)
                tech.append(("VWAP alinhado", price_ok, px, f"{'>' if side=='BUY' else '<'} {vwap:.2f}", "Confirmação de direção pelo preço vs VWAP"))
        except Exception:
            pass
    if vol_ratio is not None:
        tech.append((f"Volume {period}", vol_ratio >= min_vol, vol_ratio, f">={min_vol}", "Intensidade de volume intradiário"))
    tech.append(("Score do setup", float(score) >= float(min_score), score, f">={min_score}", "Qualidade agregada do sinal"))
    if atr is not None:
        atr_ok = atr > 0
        tech.append(("ATR válido", atr_ok, atr, ">0", "Volatilidade base para SL/TP"))
    macro = []
    try:
        df_h1 = utils.safe_copy_rates(symbol, mt5.TIMEFRAME_H1, 100)
        if df_h1 is not None and len(df_h1) > 50:
            close_h1 = df_h1["close"]
            emaf_h1 = close_h1.ewm(span=21, adjust=False).mean().iloc[-1]
            emas_h1 = close_h1.ewm(span=50, adjust=False).mean().iloc[-1]
            adx_h1 = utils.get_adx(df_h1) if hasattr(utils, "get_adx") else 0
            macro_ok = (side == "BUY" and emaf_h1 > emas_h1) or (side == "SELL" and emaf_h1 < emas_h1)
            macro.append(("H1 alinhado", macro_ok, "UP" if emaf_h1 > emas_h1 else "DOWN", side, "Confirmação multi-timeframe"))
            override_adx = float(getattr(config, "MACRO_OVERRIDE_ADX", 30) or 30)
            macro.append(("ADX H1 override", adx_h1 >= override_adx if not macro_ok else True, adx_h1, f">={override_adx}", "Override de macro se ADX forte"))
    except Exception:
        pass
    liqui = []
    try:
        ok_liq = utils.check_liquidity(symbol)
        liqui.append(("Liquidez projetada", bool(ok_liq), "OK" if ok_liq else "Baixa", ">= limiar", "Capacidade de execução sem impacto excessivo"))
    except Exception:
        liqui.append(("Liquidez projetada", False, "N/A", ">= limiar", "Dados indisponíveis"))
    oper = []
    try:
        no_entry = getattr(config, "FRIDAY_NO_ENTRY_AFTER", getattr(config, "NO_ENTRY_AFTER", "17:35")) if now.weekday() == 4 else getattr(config, "NO_ENTRY_AFTER", "17:35")
        ok_time = True
        try:
            ok_time = now.time() < datetime.strptime(no_entry, "%H:%M").time()
        except Exception:
            pass
        oper.append(("Janela de horário", ok_time, now.strftime("%H:%M"), f"< {no_entry}", "Restrições de entrada por horário"))
    except Exception:
        pass
    try:
        if BOT_CONNECTED and "daily_trades_per_symbol" in globals():
            count = int(daily_trades_per_symbol.get(symbol, 0))
            oper.append(("Limite diário", count < 4, count, "< 4", "Controle de tentativas por ativo"))
    except Exception:
        pass
    noticias = []
    try:
        is_blackout, reason = news_filter.check_news_blackout(symbol)
        noticias.append(("Sem blackout de notícias", not is_blackout, "OK" if not is_blackout else reason, "—", "Filtro de eventos macro relevantes"))
    except Exception:
        noticias.append(("Sem blackout de notícias", True, "N/A", "—", "Dados indisponíveis"))
    return {"Tecnica": tech, "Macro": macro, "Liquidez": liqui, "Operacional": oper, "Noticias": noticias}
symbols_list = _get_symbols_list()
if symbols_list:
    c1, c2 = st.columns([2, 1])
    with c1:
        selected_symbol = st.selectbox("Ativo", symbols_list)
    with c2:
        selected_side = st.radio("Lado", ["BUY", "SELL"], horizontal=True)
    crit = _compute_criteria(selected_symbol, selected_side)
    g1, g2 = st.columns(2)
    with g1:
        st.markdown("### 🎯 Técnica")
        for name, ok, value, threshold, tip in crit["Tecnica"]:
            st.markdown(_badge_row(name, ok, value, threshold, tip), unsafe_allow_html=True)
        st.markdown("### 🌎 Macro")
        for name, ok, value, threshold, tip in crit["Macro"]:
            st.markdown(_badge_row(name, ok, value, threshold, tip), unsafe_allow_html=True)
        st.markdown("### 💧 Liquidez")
        for name, ok, value, threshold, tip in crit["Liquidez"]:
            st.markdown(_badge_row(name, ok, value, threshold, tip), unsafe_allow_html=True)
    with g2:
        st.markdown("### 🛡️ Operacional")
        for name, ok, value, threshold, tip in crit["Operacional"]:
            st.markdown(_badge_row(name, ok, value, threshold, tip), unsafe_allow_html=True)
        st.markdown("### 📰 Notícias")
        for name, ok, value, threshold, tip in crit["Noticias"]:
            st.markdown(_badge_row(name, ok, value, threshold, tip), unsafe_allow_html=True)
st.markdown("---")
st.subheader("🛠️ Simulador de Diagnóstico (Tempo Real)")
with st.expander("Clique aqui para forçar um check técnico agora"):
    # Lista de ativos para monitorar
    lista_ativos = getattr(config, 'ELITE_SYMBOLS', ['WIN$N', 'WDO$N', 'PETR4', 'VALE3'])
    if isinstance(lista_ativos, dict): lista_ativos = list(lista_ativos.keys())
    
    cols_diag = st.columns(3)
    col_counter = 0
    for ativo in lista_ativos:
        with cols_diag[col_counter % 3]:
            ind = _get_indicators(ativo)
            emaf = ind.get("ema_fast")
            emas = ind.get("ema_slow")
            lado = "BUY" if (emaf is not None and emas is not None and emaf > emas) else "SELL"
            now = datetime.now()
            hour = now.hour
            if hour < 12:
                min_vol = 1.2
            elif 12 <= hour < 14:
                min_vol = float(getattr(config, "LUNCH_MIN_VOLUME_RATIO", 0.5) or 0.5)
            else:
                min_vol = 0.8
            rsi_val = ind.get("rsi")
            adx_val = ind.get("adx")
            vol_ratio = ind.get("volume_ratio")
            ok_liq = False
            try:
                ok_liq = bool(utils.check_liquidity(ativo))
            except Exception:
                ok_liq = False
            ok_rsi = (rsi_val is not None) and (30 <= float(rsi_val) <= 70)
            ok_adx = (adx_val is not None) and (float(adx_val) >= 20)
            ok_vol = (vol_ratio is not None) and (float(vol_ratio) >= float(min_vol))
            liq_html = _badge_row("Liquidez", ok_liq, "OK" if ok_liq else "Baixa", ">= limiar", "Capacidade de execução")
            rsi_html = _badge_row("RSI (14)", ok_rsi, rsi_val, "30-70", "Equilíbrio de momentum")
            adx_html = _badge_row("ADX", ok_adx, adx_val, ">=20", "Força de tendência")
            vol_html = _badge_row("Volume Ratio", ok_vol, vol_ratio, f">={min_vol}", "Intensidade intradiária")
            checks_total = 4
            checks_ok = sum([ok_liq, ok_rsi, ok_adx, ok_vol])
            erros = max(0, checks_total - checks_ok)
            aprovado = erros == 0
            cor_fundo = "#d4edda" if aprovado else "#fff3cd"
            if erros > 0: cor_fundo = "#f8d7da"
            cor_texto = "#155724" if aprovado else "#856404"
            if erros > 0: cor_texto = "#721c24"
            icon_header = "✅ OK" if aprovado else f"⚠️ {erros} FALHA(S)"
            html_card = f"""
            <div style="border: 1px solid #ddd; border-radius: 8px; margin-bottom: 10px; background: white; overflow: hidden;">
                <div style="background-color: {cor_fundo}; padding: 6px 10px; border-bottom: 1px solid {cor_texto};">
                    <div style="color: {cor_texto}; font-weight: 700; font-size: 0.95rem; display: flex; justify-content: space-between;">
                        <span>{ativo} • {lado}</span>
                        <span>{icon_header}</span>
                    </div>
                </div>
                <div style="padding: 10px;">
                    <div style="font-weight:600;color:#0f3460;margin:6px 0;">Liquidez</div>
                    {liq_html}
                    <div style="font-weight:600;color:#0f3460;margin:10px 0 6px;">RSI</div>
                    {rsi_html}
                    <div style="font-weight:600;color:#0f3460;margin:10px 0 6px;">ADX</div>
                    {adx_html}
                    <div style="font-weight:600;color:#0f3460;margin:10px 0 6px;">Volume</div>
                    {vol_html}
                </div>
            </div>
            """
            st.html(html_card)
        col_counter += 1

# ===========================
# SEÇÃO: QUADRO DE TAREFAS (Cards)
# ===========================
st.markdown("---")
st.markdown('<div class="section-header">📋 Quadro de Tarefas</div>', unsafe_allow_html=True)

def _seed_tasks():
    return [
        {"id": "tk_liquidez", "titulo": "Checar liquidez Top 15", "status": "pending", "categoria": "Operacional", "desc": "Validar volume projetado e profundidade do book"},
        {"id": "tk_ml", "titulo": "Treinar ensemble ML diário", "status": "completed", "categoria": "Técnica", "desc": "Atualizar modelo com trades recentes"},
        {"id": "tk_heat", "titulo": "Atualizar heatmap setorial", "status": "pending", "categoria": "Macro", "desc": "Recalcular exposição e correlações"},
        {"id": "tk_rel", "titulo": "Gerar relatório parcial", "status": "completed", "categoria": "Relatórios", "desc": "Resumo de execução e rejeições do dia"},
        {"id": "tk_alert", "titulo": "Verificar alertas críticos", "status": "pending", "categoria": "Operacional", "desc": "Circuit breaker / Profit lock / VIX switching"},
    ]

if "task_board" not in st.session_state:
    st.session_state["task_board"] = _seed_tasks()
if "task_view" not in st.session_state:
    st.session_state["task_view"] = "complete"  # complete | pending

col_tv1, col_tv2 = st.columns([1, 1])
with col_tv1:
    st.caption("Modo de visualização")
    st.session_state["task_view"] = st.segmented_control(
        "Mostrar",
        options=["Completa", "Pendentes"],
        default="Completa",
        key="task_view_segment",
    )
with col_tv2:
    st.caption("Resumo")
    total_tarefas = len(st.session_state["task_board"])
    pendentes = sum(1 for t in st.session_state["task_board"] if t["status"] == "pending")
    concluidas = total_tarefas - pendentes
    st.metric("Total", total_tarefas)
    st.metric("Pendentes", pendentes)

# Filtro dinâmico (sem recarregar a página além do rerun padrão)
mostrar_pendentes = (st.session_state["task_view"] == "Pendentes")
tarefas_exibidas = [t for t in st.session_state["task_board"] if (t["status"] == "pending") or not mostrar_pendentes]

# Layout responsivo em cards
cols = st.columns(3)
for idx, t in enumerate(tarefas_exibidas):
    col = cols[idx % 3]
    bg = "#fef9e7" if t["status"] == "pending" else "#e8f5e9"
    border = "#f39c12" if t["status"] == "pending" else "#2ecc71"
    icon = "⏳" if t["status"] == "pending" else "✅"
    with col:
        st.markdown(
            f"""
            <div style="background:{bg};border-left:4px solid {border};border-radius:10px;padding:12px;margin-bottom:12px;">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <div style="font-weight:700;color:#1a1a2e;">{icon} {t['titulo']}</div>
                    <div style="font-size:0.8rem;color:#636e72;">{t['categoria']}</div>
                </div>
                <div style="margin-top:6px;color:#2d3436;">{t['desc']}</div>
                <div style="margin-top:8px;">
                    <span style="font-size:0.8rem;padding:4px 8px;border-radius:12px;background:{border}20;color:{border};">
                        {('Pendente' if t['status']=='pending' else 'Concluída')}
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
# ===========================
# FOOTER
# ===========================
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns(3)
with footer_col1:
    st.caption(f"🕐 Última atualização: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
with footer_col2:
    st.caption("🏛️ XP3 PRO B3 Institutional Dashboard v6.0")
with footer_col3:
    st.caption(f"© 2025 xAI Inspired")
if auto_refresh:
    if time.time() >= st.session_state.get("pause_refresh_until", 0.0):
        time.sleep(refresh_interval)
        st.rerun()
