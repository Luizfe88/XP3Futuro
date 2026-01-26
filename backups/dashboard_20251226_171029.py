# dashboard_xp3.py - XP3 PRO B3 DASHBOARD v4.0
"""
🚀 XP3 PRO B3 DASHBOARD - Streamlit Real-Time
✅ Compatível com bot.py v3.0+
✅ Métricas em tempo real
✅ Gráficos interativos
✅ Controle do bot
✅ Machine Learning insights
"""

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

# ===========================
# IMPORTAÇÕES DO BOT
# ===========================
try:
    # Adiciona o diretório do bot ao path se necessário
    if '.' not in sys.path:
        sys.path.insert(0, '.')
    
    import config
    import utils
    
    # Tenta importar do bot - com fallbacks
    try:
        from bot import (
            bot_state,
            position_open_times,
            last_close_time,
            trading_paused,
            equity_inicio_dia,
            daily_max_equity,
            get_market_status,
            daily_trades_per_symbol
        )
        BOT_CONNECTED = True
    except ImportError as e:
        st.warning(f"⚠️ Algumas funções do bot não disponíveis: {e}")
        BOT_CONNECTED = False
        
    try:
        from ml_optimizer import ml_optimizer
        ML_AVAILABLE = True
    except ImportError:
        ML_AVAILABLE = False
        st.warning("⚠️ ML Optimizer não disponível")
        
except ImportError as e:
    st.error(f"❌ Erro ao importar módulos: {e}")
    st.stop()

# ===========================
# CONFIGURAÇÃO DA PÁGINA
# ===========================
st.set_page_config(
    page_title="XP3 B3 Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #00ff88 0%, #00d4ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .profit-positive {
        color: #00ff88;
        font-weight: bold;
    }
    .profit-negative {
        color: #ff4444;
        font-weight: bold;
    }
    .status-active {
        background-color: #00ff88;
        color: black;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: bold;
    }
    .status-paused {
        background-color: #ff9800;
        color: black;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: bold;
    }
    .status-weekend {
        background-color: #666;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: bold;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
    }
</style>
""", unsafe_allow_html=True)

# ===========================
# FUNÇÕES AUXILIARES
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
        st.error(f"Erro ao conectar MT5: {e}")
    return None



@st.cache_data(ttl=5)
def load_positions():
    """Carrega posições abertas"""
    try:
        with utils.mt5_lock:
            positions = mt5.positions_get() or []
        
        if not positions:
            return pd.DataFrame()
        
        data = []
        for pos in positions:
            side = "BUY" if pos.type == mt5.POSITION_TYPE_BUY else "SELL"
            
            # Calcula tempo aberto buscando direto do MT5 (mais confiável)
            # p.time é o timestamp de abertura registrado na corretora
            abertura_ts = pos.time  
            agora_ts = time.time()

            segundos_decorridos = agora_ts - abertura_ts
            minutos_decorridos = segundos_decorridos / 60

            if minutos_decorridos < 60:
                time_str = f"{int(minutos_decorridos)}m"
            else:
                horas = int(minutos_decorridos // 60)
                mins = int(minutos_decorridos % 60)
                time_str = f"{horas}h {mins}m"
   
            
            # Calcula P&L percentual
            if side == "BUY":
                pnl_pct = ((pos.price_current - pos.price_open) / pos.price_open) * 100
            else:
                pnl_pct = ((pos.price_open - pos.price_current) / pos.price_open) * 100
            
            data.append({
                "Ticket": pos.ticket,
                "Símbolo": pos.symbol,
                "Lado": side,
                "Volume": f"{pos.volume:.2f}",
                "Entrada": pos.price_open,
                "Atual": pos.price_current,
                "SL": pos.sl if pos.sl > 0 else "-",
                "TP": pos.tp if pos.tp > 0 else "-",
                "P&L R$": pos.profit,
                "P&L %": pnl_pct,
                "Tempo": time_str
            })
        
        return pd.DataFrame(data)
    except Exception as e:
        st.error(f"Erro ao carregar posições: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=30)
def load_trade_history():
    """Carrega histórico de trades do ML"""
    try:
        if ML_AVAILABLE and hasattr(ml_optimizer, 'history_file'):
            if Path(ml_optimizer.history_file).exists():
                with open(ml_optimizer.history_file, 'r') as f:
                    data = json.load(f)
                
                if data:
                    df = pd.DataFrame(data)
                    if 'timestamp' in df.columns:
                        df["timestamp"] = pd.to_datetime(df["timestamp"])
                    return df
    except Exception as e:
        st.warning(f"Não foi possível carregar histórico ML: {e}")
    
    return pd.DataFrame()

@st.cache_data(ttl=5)
def load_today_trades():
    """Carrega trades do arquivo TXT de hoje"""
    try:
        filename = f"trades_log_{datetime.now().strftime('%Y-%m-%d')}.txt"
        
        if not os.path.exists(filename):
            return pd.DataFrame()
        
        data = []
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line in lines[2:]:  # Pula header
            if line.strip() and not line.startswith('-'):
                parts = line.split('|')
                if len(parts) >= 8:
                    try:
                        data.append({
                            'Timestamp': parts[0].strip(),
                            'Tipo': parts[1].strip(),
                            'Símbolo': parts[2].strip(),
                            'Lado': parts[3].strip(),
                            'Volume': parts[4].strip(),
                            'Preço': parts[5].strip(),
                            'P&L': parts[6].strip(),
                            'Motivo': parts[7].strip() if len(parts) > 7 else ''
                        })
                    except:
                        continue
        
        return pd.DataFrame(data)
    except Exception as e:
        return pd.DataFrame()

def format_number(value, prefix="", suffix="", decimals=2):
    """Formata números com cores"""
    color = "profit-positive" if value >= 0 else "profit-negative"
    sign = "+" if value > 0 else ""
    return f'<span class="{color}">{prefix}{sign}{value:,.{decimals}f}{suffix}</span>'

def get_bot_status():
    """Retorna status do bot de forma segura"""
    if not BOT_CONNECTED:
        return {
            "status": "DISCONNECTED",
            "message": "Dashboard Standalone",
            "emoji": "⚠️",
            "paused": False
        }
    
    try:
        market_status = get_market_status()
        
        return {
            "status": market_status["status"],
            "message": market_status["message"],
            "emoji": market_status["emoji"],
            "paused": trading_paused if 'trading_paused' in globals() else False,
            "detail": market_status.get("detail", ""),
            "countdown": market_status.get("countdown", None)
        }
    except Exception as e:
        return {
            "status": "ERROR",
            "message": f"Erro: {e}",
            "emoji": "❌",
            "paused": False
        }

# ===========================
# HEADER
# ===========================
st.markdown('<div class="main-header">🚀 XP3 PRO B3 DASHBOARD v4.0</div>', unsafe_allow_html=True)

# ===========================
# SIDEBAR - CONTROLES
# ===========================
with st.sidebar:
    st.image("https://via.placeholder.com/200x80/667eea/ffffff?text=XP3+B3", width='stretch')
    
    st.markdown("---")
    st.subheader("⚙️ Controles")
    
    # Auto-refresh
    auto_refresh = st.checkbox("🔄 Auto-refresh", value=True)
    refresh_interval = st.slider("Intervalo (segundos)", 3, 30, 5)
    
    if st.button("🔄 Atualizar Agora", width='stretch'):
        st.cache_data.clear()
        st.rerun()
    
    st.markdown("---")
    
    # Status do bot
    bot_status = get_bot_status()
    
    if bot_status["status"] == "WEEKEND":
        status_class = "status-weekend"
    elif bot_status["paused"]:
        status_class = "status-paused"
    elif bot_status["status"] in ["OPEN", "NO_NEW_ENTRIES"]:
        status_class = "status-active"
    else:
        status_class = "status-paused"
    
    status_text = f"{bot_status['emoji']} {bot_status['message']}"
    st.markdown(f'<div class="{status_class}">{status_text}</div>', unsafe_allow_html=True)
    
    if bot_status.get("detail"):
        st.info(bot_status["detail"])
    
    if bot_status.get("countdown"):
        st.metric("Countdown", bot_status["countdown"])
    
    st.markdown("---")
    
    # Informações do sistema
    st.subheader("💻 Sistema")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Python", f"{sys.version_info.major}.{sys.version_info.minor}")
    with col2:
        terminal_status = "✅" if mt5.terminal_info() else "❌"
        st.metric("MT5", terminal_status)
    
    # ML Status
    if ML_AVAILABLE:
        ml_trades = len(ml_optimizer.history) if hasattr(ml_optimizer, 'history') else 0
        epsilon = getattr(ml_optimizer, 'epsilon', 0.0)
        st.metric("ML Trades", ml_trades)
        st.metric("Epsilon", f"{epsilon:.3f}")

# ===========================
# MAIN CONTENT
# ===========================

# Carrega dados
acc = load_account_info()
positions_df = load_positions()

if not acc:
    st.error("❌ Não foi possível conectar ao MT5. Verifique a conexão.")
    st.stop()

# ===========================
# ROW 1 - MÉTRICAS PRINCIPAIS
# ===========================
st.subheader("💰 Visão Geral da Conta")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "Balance",
        f"R${acc['balance']:,.2f}",
        delta=f"{acc['profit']:+,.2f}"
    )

with col2:
    st.metric(
        "Equity",
        f"R${acc['equity']:,.2f}",
        delta=f"{(acc['equity'] - acc['balance']):+,.2f}"
    )

with col3:
    pnl_pct = (acc['profit'] / acc['balance'] * 100) if acc['balance'] > 0 else 0
    st.metric(
        "P&L Hoje",
        f"R${acc['profit']:,.2f}",
        delta=f"{pnl_pct:+.2f}%"
    )

with col4:
    st.metric(
        "Margem Livre",
        f"R${acc['free_margin']:,.2f}",
        delta=f"{acc['margin_level']:.1f}%"
    )

with col5:
    max_positions = getattr(config, 'MAX_SYMBOLS', 10)
    st.metric(
        "Posições",
        len(positions_df),
        delta=f"{max_positions - len(positions_df)} livres"
    )

# ===========================
# ROW 2 - LUCRO REALIZADO
# ===========================
st.markdown("---")

try:
    lucro_realizado, qtd_trades = utils.calcular_lucro_realizado_txt()
    lucro_flutuante = positions_df['P&L R$'].sum() if not positions_df.empty else 0
    lucro_total = lucro_realizado + lucro_flutuante
    
    col_real, col_flut, col_total = st.columns(3)
    
    with col_real:
        st.metric(
            "💰 Realizado (no bolso)",
            f"R${lucro_realizado:,.2f}",
            delta=f"{qtd_trades} trades"
        )
    
    with col_flut:
        st.metric(
            "📈 Flutuante (aberto)",
            f"R${lucro_flutuante:+,.2f}",
            delta=f"{len(positions_df)} posições"
        )
    
    with col_total:
        pnl_total_pct = (lucro_total / acc['balance']) * 100 if acc['balance'] > 0 else 0
        st.metric(
            "🏆 TOTAL DO DIA",
            f"R${lucro_total:+,.2f}",
            delta=f"{pnl_total_pct:+.2f}%"
        )
except Exception as e:
    st.warning(f"Não foi possível calcular lucro realizado: {e}")

# ===========================
# ROW 3 - POSIÇÕES ABERTAS (VERSÃO FINAL - SEM ERRO DE FORMATAÇÃO)
# ===========================
st.markdown("---")
st.subheader(f"📍 Posições Abertas ({len(positions_df)})")

if not positions_df.empty:
    # === FUNÇÃO PARA OBTER ATR E CALCULAR STATUS ===
    def get_atr_from_bridge(symbol):
        try:
            if os.path.exists("bot_bridge.json"):
                with open("bot_bridge.json", "r", encoding="utf-8") as f:
                    bridge = json.load(f)
                return bridge.get("indicators", {}).get(symbol, {}).get("atr", 0.01)
        except:
            pass
        return 0.01

    def get_status(row):
        try:
            atr = get_atr_from_bridge(row["Símbolo"])
            if atr <= 0:
                atr = 0.01

            profit_points = (
                row["Atual"] - row["Entrada"] if row["Lado"] == "BUY"
                else row["Entrada"] - row["Atual"]
            )
            profit_atr = abs(profit_points / atr)

            if profit_atr >= 2.5:
                return "🔒 Trailing"
            elif profit_atr >= 1.0:
                return "🛡️ Breakeven"
            else:
                return "⏳ Aguardando"
        except:
            return "❓ Indisponível"

    # Prepara o DataFrame
    positions_display = positions_df.copy()

    # Converte Volume para float (para formatação correta)
    positions_display["Volume"] = pd.to_numeric(positions_display["Volume"], errors='coerce').fillna(0)

    # Adiciona Status
    positions_display["Status"] = positions_display.apply(get_status, axis=1)

    # Ordem das colunas - exatamente como no painel do bot
    cols_order = ["Símbolo", "Lado", "Volume", "Entrada", "Atual", "SL", "TP", "P&L R$", "P&L %", "Status"]
    positions_display = positions_display[cols_order]

    # === ESTILOS ===
    def color_pnl_money(val):
        try:
            v = float(val)
            return "color: #00ff88; font-weight: bold" if v >= 0 else "color: #ff4444; font-weight: bold"
        except:
            return ""

    def color_pnl_pct(val):
        try:
            v = float(val)
            return "color: #00ff88; font-weight: bold" if v >= 0 else "color: #ff4444; font-weight: bold"
        except:
            return ""

    def color_status(row):
        status = row["Status"]
        if "Trailing" in status:
            color = "#00ff88"
        elif "Breakeven" in status:
            color = "#00d4ff"
        elif "Aguardando" in status:
            color = "#ff9800"
        else:
            color = "#ff4444"
        return [f"color: {color}; font-weight: bold" if col == "Status" else "" for col in row.index]

    # Formatação SEGURA - só em colunas numéricas garantidas
    format_dict = {
        "Entrada": "{:.2f}",
        "Atual": "{:.2f}",
        "SL": "{:.2f}",
        "TP": "{:.2f}",
        "P&L R$": "R$ {:+.2f}",
        "P&L %": "{:+.2f}%",
        "Volume": "{:.0f}",  # Agora é seguro porque convertemos para float acima
    }

    styled_df = (
        positions_display
        .style
        .map(color_pnl_money, subset=["P&L R$"])
        .map(color_pnl_pct, subset=["P&L %"])
        .apply(color_status, axis=1)
        .format(format_dict)
    )

    st.dataframe(
        styled_df,
        use_container_width=True,
        hide_index=True,
        height=450
    )
else:
    st.info("✅ Nenhuma posição aberta no momento")

# ===========================
# ROW 4 - TABS AVANÇADAS
# ===========================
st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "🔥 Top Pares",
    "📜 Histórico Hoje",
    "🤖 Machine Learning",
    "⚙️ Configurações"
])

# TAB 1: TOP PARES
with tab1:
    st.subheader("🔥 Ativos Monitorados - TOP 15 Elite")
    
    bridge_data = None
    if os.path.exists("bot_bridge.json"):
        try:
            with open("bot_bridge.json", "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    bridge_data = json.loads(content)
        except Exception as e:
            st.error(f"Erro ao ler ponte: {e}")
    
    if not bridge_data or not bridge_data.get("top15"):
        st.info("🔄 Aguardando sincronização com o bot... (o bot precisa estar rodando)")
        st.caption("Dica: Verifique se o bot.py está executando e atualizando bot_bridge.json")
    else:
        df_top = pd.DataFrame(bridge_data["top15"])
        
        # Ordena colunas bonitas
        cols = ["rank", "symbol", "score", "direction", "rsi", "atr_pct", "price", "sector", "status"]
        df_top = df_top[cols]
        df_top.columns = ["RK", "Símbolo", "Score", "Direção", "RSI", "ATR%", "Preço", "Setor", "Status"]
        
        # Estilo
        def style_status(val):
            if "ABERTO" in val:
                return "color: #00ff88; font-weight: bold"
            elif "PRONTO" in val:
                return "color: #00d4ff; font-weight: bold"
            else:
                return "color: #ff9800"
        
        def style_direction(val):
            if "LONG" in val:
                return "color: #00ff88; font-weight: bold"
            elif "SHORT" in val:
                return "color: #ff4444; font-weight: bold"
            else:
                return ""
        
        styled = (
            df_top.style
            .format({
                "Score": "{:.1f}",
                "RSI": "{:.1f}",
                "ATR%": "{:.2f}",
                "Preço": "{:.2f}"
            })
            .map(style_status, subset=["Status"])
            .map(style_direction, subset=["Direção"])
        )
        
        st.dataframe(styled, use_container_width=True, hide_index=True, height=500)
        
        st.caption(f"🕐 Última atualização: {bridge_data.get('timestamp', 'N/A')}")

# TAB 2: HISTÓRICO HOJE
with tab2:
    st.subheader("📜 Trades de Hoje")
    
    today_trades = load_today_trades()
    
    if not today_trades.empty:
        # Filtros
        col_f1, col_f2 = st.columns(2)
        
        with col_f1:
            tipo_filter = st.multiselect(
                "Filtrar Tipo",
                options=today_trades["Tipo"].unique().tolist(),
                default=[]
            )
        
        with col_f2:
            symbol_filter = st.multiselect(
                "Filtrar Símbolos",
                options=today_trades["Símbolo"].unique().tolist(),
                default=[]
            )
        
        # Aplica filtros
        filtered_df = today_trades.copy()
        
        if tipo_filter:
            filtered_df = filtered_df[filtered_df["Tipo"].isin(tipo_filter)]
        
        if symbol_filter:
            filtered_df = filtered_df[filtered_df["Símbolo"].isin(symbol_filter)]
        
        # Estatísticas
        st.subheader("📊 Estatísticas do Dia")
        
        stats_col1, stats_col2, stats_col3 = st.columns(3)
        
        with stats_col1:
            st.metric("Total Operações", len(filtered_df))
        
        with stats_col2:
            entradas = len(filtered_df[filtered_df["Tipo"] == "ENTRADA"])
            saidas = len(filtered_df[filtered_df["Tipo"] == "SAÍDA"])
            st.metric("Entradas", entradas)
        
        with stats_col3:
            st.metric("Saídas", saidas)
        
        # Tabela
        st.dataframe(filtered_df, width='stretch', height=400)
    else:
        st.info("📭 Nenhum trade registrado hoje")

# TAB 3: MACHINE LEARNING
with tab3:
    st.subheader("🤖 Machine Learning Insights")
    
    if ML_AVAILABLE:
        try:
            col_ml1, col_ml2, col_ml3 = st.columns(3)
            
            with col_ml1:
                trades_count = len(ml_optimizer.history) if hasattr(ml_optimizer, 'history') else 0
                st.metric("Trades Treinados", trades_count)
            
            with col_ml2:
                epsilon = getattr(ml_optimizer, 'epsilon', 0.0)
                st.metric("Taxa Exploração", f"{epsilon:.2%}")
            
            with col_ml3:
                q_exists = os.path.exists(getattr(ml_optimizer, 'qtable_file', 'qtable.npy'))
                st.metric("Q-Table", "Carregada" if q_exists else "Nova")
            
            # Histórico ML
            ml_history = load_trade_history()
            
            if not ml_history.empty and 'pnl_pct' in ml_history.columns:
                st.markdown("---")
                st.subheader("📊 Performance ML")
                
                # Gráfico de P&L acumulado
                ml_history['cumulative_pnl'] = ml_history['pnl_pct'].cumsum()
                
                fig_ml = go.Figure()
                fig_ml.add_trace(go.Scatter(
                    x=ml_history.index,
                    y=ml_history['cumulative_pnl'],
                    mode='lines',
                    name='P&L Acumulado',
                    line=dict(color='#00d4ff', width=2)
                ))
                
                fig_ml.update_layout(
                    template="plotly_dark",
                    height=300,
                    title="P&L Acumulado (ML)",
                    xaxis_title="Trade #",
                    yaxis_title="P&L %"
                )
                
                st.plotly_chart(fig_ml, width='stretch')
        except Exception as e:
            st.error(f"Erro ao carregar ML: {e}")
    else:
        st.info("ML Optimizer não disponível")

# TAB 4: CONFIGURAÇÕES
with tab4:
    st.subheader("⚙️ Configurações do Bot")
    
    col_cfg1, col_cfg2 = st.columns(2)
    
    with col_cfg1:
        st.markdown("### 🎯 Risk Management")
        risk_info = f"""
```
Risk por Trade: {config.RISK_PER_TRADE_PCT:.1%}
Max Posições: {config.MAX_SYMBOLS}
Max DD Diário: {config.MAX_DAILY_DRAWDOWN_PCT:.1%}
SL ATR Mult: {config.SL_ATR_MULTIPLIER}x
```
        """
        st.markdown(risk_info)
        
        st.markdown("### 📊 Trading")
        trading_info = f"""
```
Pares Elite: {len(config.ELITE_SYMBOLS)}
Magic Number: 2026
Horário: {config.TRADING_START} - {config.CLOSE_ALL_BY}
```
        """
        st.markdown(trading_info)
    
    with col_cfg2:
        st.markdown("### 🤖 Machine Learning")
        ml_status = "✅" if ML_AVAILABLE else "❌"
        ml_info = f"""
```
Q-Learning: {ml_status}
Walk-Forward: ✅
Auto-adjust: ✅
```
        """
        st.markdown(ml_info)
        
        st.markdown("### 💬 Telegram")
        tg_enabled = getattr(config, 'ENABLE_TELEGRAM_NOTIF', False)
        tg_status = "✅" if tg_enabled else "❌"
        tg_info = f"""
```
Notificações: {tg_status}
Relatório EOD: {config.EOD_REPORT_ENABLED}
```
        """
        st.markdown(tg_info)

# ===========================
# FOOTER & AUTO-REFRESH
# ===========================
st.markdown("---")

footer_col1, footer_col2, footer_col3 = st.columns([1, 2, 1])

with footer_col1:
    st.caption(f"🕐 Última atualização: {datetime.now().strftime('%H:%M:%S')}")

with footer_col2:
    st.caption("🚀 XP3 PRO B3 BOT v4.0 | Dashboard by Streamlit")

with footer_col3:
    if auto_refresh:
        st.caption(f"🔄 Auto-refresh: {refresh_interval}s")
        time.sleep(refresh_interval)
        st.rerun()