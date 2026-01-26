# telegram_handler.py

import telebot
import logging
from datetime import datetime
import MetaTrader5 as mt5
import config
from utils import send_telegram_message  # opcional, se quiser usar sua função
from news_filter import get_next_high_impact_event, check_news_blackout, get_upcoming_events

logger = logging.getLogger("telegram")

# Só cria o bot se Telegram estiver habilitado
if getattr(config, "ENABLE_TELEGRAM_NOTIF", False):
    bot = telebot.TeleBot(config.TELEGRAM_BOT_TOKEN)
    CHAT_ID = config.TELEGRAM_CHAT_ID  # Para envios automáticos
else:
    bot = None
    CHAT_ID = None

# ==================== HANDLERS ====================

@bot.message_handler(commands=['start', 'help'])
def handle_help(message):
    help_text = """
🤖 <b>XP3 PRO - Comandos Disponíveis</b>

📊 <b>Informações</b>
/status         → Status do bot e conexão
/lucro          → Lucro do dia e posições
/health         → Latência, memória e status do sistema
/proximoevento  → Próximo evento econômico importante
/blackout ou /news → Status de blackout por notícia

ℹ️ Bot opera automaticamente na B3.
    """
    bot.reply_to(message, help_text, parse_mode="HTML")


@bot.message_handler(commands=['status'])
def handle_status(message):
    if not mt5.terminal_info() or not mt5.terminal_info().connected:
        status = "🔴 <b>MT5 DESCONECTADO</b>"
    else:
        acc = mt5.account_info()
        balance = acc.balance if acc else 0
        equity = acc.equity if acc else 0
        positions_count = len(mt5.positions_get() or [])
        
        status = (
            f"🤖 <b>XP3 PRO - STATUS</b>\n\n"
            f"✅ <b>Conectado ao MT5</b>\n"
            f"⏰ {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n\n"
            f"💰 Balance: R$ {balance:,.2f}\n"
            f"📈 Equity:  R$ {equity:,.2f}\n"
            f"📊 Posições abertas: {positions_count}"
        )
    
    bot.reply_to(message, status, parse_mode="HTML")


@bot.message_handler(commands=['lucro'])
def handle_lucro(message):
    acc = mt5.account_info()
    if not acc:
        bot.reply_to(message, "❌ Não conectado ao MT5")
        return
    
    profit_today = acc.profit
    positions = mt5.positions_get() or []
    
    msg = (
        f"📊 <b>RESUMO DO DIA</b>\n\n"
        f"💰 Lucro realizado + flutuante: <b>{profit_today:+.2f} R$</b>\n"
        f"📈 Posições abertas: {len(positions)}\n"
    )
    
    if positions:
        msg += "\n<b>Posições atuais:</b>\n"
        for p in positions[:8]:
            emoji = "🟢" if p.profit >= 0 else "🔴"
            msg += f"{emoji} {p.symbol} | Vol: {p.volume} | P&L: {p.profit:+.2f} R$\n"
    
    bot.reply_to(message, msg, parse_mode="HTML")


@bot.message_handler(commands=['proximoevento'])
def handle_proximoevento(message):
    event_msg = get_next_high_impact_event()
    emoji = "🔴" if "em" in event_msg.lower() and "min" in event_msg.lower() else "🟢"
    full_msg = f"{emoji} <b>PRÓXIMO EVENTO</b>\n\n{event_msg}"
    bot.reply_to(message, full_msg, parse_mode="HTML")


@bot.message_handler(commands=['blackout', 'news'])
def handle_blackout(message):
    blocked, reason = check_news_blackout()
    upcoming = get_upcoming_events(hours_ahead=8)
    
    if blocked:
        status = f"🚫 <b>BOT EM BLACKOUT</b>\n\n{reason}\n\nEntradas bloqueadas até passar o evento."
    else:
        if upcoming:
            ev = upcoming[0]
            mins = int((ev["time"] - datetime.now()).total_seconds() / 60)
            emoji = "🔴" if ev["impact"] == "High" else "🟡"
            status = (
                f"✅ <b>TRADING LIBERADO</b>\n\n"
                f"{emoji} Próximo: <b>{ev['title']}</b>\n"
                f"⏰ Em {mins} minutos ({ev['impact']} impacto)"
            )
        else:
            status = "✅ <b>TRADING LIBERADO</b>\n\nSem eventos nas próximas 8 horas."
    
    bot.reply_to(message, status, parse_mode="HTML")


@bot.message_handler(commands=['health'])
def handle_health(message):
    """
    Retorna status de saúde do sistema:
    - Latência com a corretora
    - Status da conexão MT5
    - Uso de memória
    """
    import time as time_module
    
    health_info = []
    
    # 1. Status MT5
    terminal = mt5.terminal_info()
    if terminal and terminal.connected:
        health_info.append("✅ <b>MT5:</b> Conectado")
        
        # Latência (tempo de resposta do tick)
        start = time_module.time()
        tick = mt5.symbol_info_tick("PETR4")
        latency_ms = (time_module.time() - start) * 1000
        
        if latency_ms < 100:
            health_info.append(f"🟢 <b>Latência:</b> {latency_ms:.0f}ms (excelente)")
        elif latency_ms < 300:
            health_info.append(f"🟡 <b>Latência:</b> {latency_ms:.0f}ms (ok)")
        else:
            health_info.append(f"🔴 <b>Latência:</b> {latency_ms:.0f}ms (lenta)")
        
        # Conta
        acc = mt5.account_info()
        if acc:
            health_info.append(f"💰 <b>Conta:</b> {acc.login}")
            health_info.append(f"🏢 <b>Corretora:</b> {acc.company}")
    else:
        health_info.append("🔴 <b>MT5:</b> DESCONECTADO")
        health_info.append("⚠️ <b>Latência:</b> N/A")
    
    # 2. Uso de memória
    try:
        import psutil
        process = psutil.Process()
        mem_mb = process.memory_info().rss / 1024 / 1024
        cpu_pct = process.cpu_percent()
        
        if mem_mb < 500:
            health_info.append(f"🟢 <b>Memória:</b> {mem_mb:.0f} MB")
        elif mem_mb < 1000:
            health_info.append(f"🟡 <b>Memória:</b> {mem_mb:.0f} MB")
        else:
            health_info.append(f"🔴 <b>Memória:</b> {mem_mb:.0f} MB (alta)")
        
        health_info.append(f"⚡ <b>CPU:</b> {cpu_pct:.1f}%")
    except ImportError:
        health_info.append("⚠️ <b>Memória:</b> psutil não instalado")
    except Exception as e:
        health_info.append(f"⚠️ <b>Memória:</b> Erro ({e})")
    
    # 3. Posições abertas
    positions = mt5.positions_get() or []
    health_info.append(f"📊 <b>Posições:</b> {len(positions)}")
    
    # 4. Timestamp
    health_info.append(f"\n⏰ {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    
    msg = "🏥 <b>HEALTH CHECK</b>\n\n" + "\n".join(health_info)
    bot.reply_to(message, msg, parse_mode="HTML")


@bot.message_handler(commands=['winrate'])
def handle_winrate(message):
    try:
        args = message.text.split()
        if len(args) < 2:
            # Win Rate Geral
            from utils import get_realtime_win_rate
            wr_data = get_realtime_win_rate(lookback_trades=50)
            
            msg = (
                f"📊 <b>WIN RATE GERAL (Últimos {wr_data['total_trades']})</b>\n\n"
                f"🎯 <b>Win Rate:</b> {wr_data['win_rate']:.1%}\n"
                f"📉 Trades: {wr_data['total_trades']} (✅{wr_data['wins']} ❌{wr_data['losses']})\n"
                f"💰 Profit Factor: {wr_data['profit_factor']:.2f}\n"
                f"⚖️ Expectativa: R$ {wr_data['expectancy']:.2f}"
            )
            bot.reply_to(message, msg, parse_mode="HTML")
            return

        symbol = args[1].upper()
        if not symbol.endswith(".SA") and len(symbol) <= 5: # Ajuste simples se user esquecer .SA
             # Mas XP3 usa tickers sem SA internamente às vezes? O padrão B3 é sem SA internamente na lógica ou com?
             # bot.py usa mt5 ticks que dependem. Assumindo input flexível.
             pass
             
        from utils import get_symbol_performance
        perf = get_symbol_performance(symbol, lookback_days=30)
        
        msg = (
            f"📊 <b>WIN RATE: {symbol} (30d)</b>\n\n"
            f"🎯 <b>Win Rate:</b> {perf['win_rate']:.1%}\n"
            f"📉 Trades: {perf['trades']} (✅{perf['wins']} ❌{perf['losses']})\n"
            f"💰 PnL Total: R$ {perf['total_pnl']:+,.2f}"
        )
        bot.reply_to(message, msg, parse_mode="HTML")
        
    except Exception as e:
        logger.error(f"Erro comando winrate: {e}")
        bot.reply_to(message, "❌ Erro ao calcular win rate.")

# ====================
# 🔮 PREDICTIVE ALERTS
# ====================

def send_predictive_alert(symbol: str, current_win_rate: float, predicted_drawdown: float = 0.0):
    """
    Envia alerta preditivo se métricas deteriorarem.
    Pode ser chamado periodicamente pelo bot principal.
    """
    if not bot: return
    
    try:
        if current_win_rate < 0.55:
            emoji = "⚠️" if current_win_rate >= 0.45 else "🚨"
            severity = "ALTA" if current_win_rate < 0.45 else "MÉDIA"
            
            msg = (
                f"{emoji} <b>ALERTA PREDITIVO DE RISCO</b>\n\n"
                f"📉 <b>Win Rate em Queda:</b> {current_win_rate:.1%}\n"
                f"⚠️ <b>Severidade:</b> {severity}\n\n"
                f"💡 <i>Sugestão: O bot ativará modo defensivo em &lt;50%</i>"
            )
            
            # Evita spam: idealmente checar último envio (simplificado aqui)
            bot.send_message(CHAT_ID, msg, parse_mode="HTML")
            
    except Exception as e:
        logger.error(f"Erro ao enviar alerta preditivo: {e}")


# Exporta o bot para uso externo
__all__ = ['bot', 'telegram_polling_thread', 'send_predictive_alert']