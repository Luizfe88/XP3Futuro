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
    
    # Formatação de tabela
    cols_to_show = ['Símbolo', 'Lado', 'Volume', 'Preço Entrada', 'Preço Atual',
                    'Stop Loss', 'P&L R$', 'P&L %', 'Exposição %', 'Tempo Aberto', 'Status']
    
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
    st.download_button(
        label="📥 Exportar Posições (CSV)",
        data=csv_positions,
        file_name=f"posicoes_abertas_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv"
    )
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
        st.download_button(
            label="📥 Baixar Excel",
            data=excel_data,
            file_name=f"trading_report_{datetime.now().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
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
    time.sleep(refresh_interval)
    st.rerun()