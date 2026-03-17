# mentorship_reporter.py
"""
🎓 SISTEMA DE MENTORIA DIÁRIA - XP3 PRO
=======================================
Orquestra o envio do relatório consolidado:
1. Relatório de Aprendizado (IA Adaptativa)
2. Relatório Didático (Justificativa de Trades e Rejeições)
"""

import logging
from datetime import datetime
from daily_learning_report import DailyLearningReport
from daily_analysis_logger import daily_logger
import config
from telegram_handler import bot as telegram_bot

logger = logging.getLogger("mentorship_reporter")

class MentorshipReporter:
    """
    Agrega dados do DailyLearningReport e DailyAnalysisLogger
    para criar a experiência de mentoria diária.
    """
    
    def __init__(self):
        self.learner = DailyLearningReport()
        
    def generate_mentorship_message(self) -> str:
        """
        Gera a mensagem consolidada da mentoria.
        """
        logger.info("🎓 Gerando Mensagem de Mentoria Diária...")
        
        # 1. Parte de Aprendizado (IA)
        # O generate_and_apply já atualiza pesos e retorna o texto formatado para Telegram
        learning_report = self.learner.generate_and_apply()
        
        # 2. Parte Didática (Performance e Filtros)
        didactic_summary = daily_logger.get_daily_rejection_summary()
        
        # 3. Consolidação
        msg = (
            "🎓 <b>MENTORIA DIÁRIA XP3</b>\t\n"
            "───────────────────────────\n\n"
            "🧠 <b>RELATÓRIO DE APRENDIZADO</b>\n"
            f"{learning_report if learning_report else 'Sem dados de aprendizado hoje.'}\n\n"
            "───────────────────────────\n"
            "📖 <b>RELATÓRIO DIDÁTICO (TRACKER)</b>\n"
            "<i>Por que alguns trades foram filtrados?</i>\n"
            "<i>Conceitos: Spread, EMA, RSI Esticado e ADX.</i>\n\n"
            f"<code>{didactic_summary}</code>\n\n"
            "💡 <i>Insight: A IA aprendeu com as rejeições de hoje para calibrar os filtros de amanhã.</i>"
        )
        
        return msg

    def send_report(self, chat_id: int = None):
        """
        Gera e envia o relatório via Telegram.
        """
        if chat_id is None:
            chat_id = config.TELEGRAM_CHAT_ID
            
        try:
            message = self.generate_mentorship_message()
            
            # Divide a mensagem se for muito longa para o Telegram (4096 chars)
            if len(message) > 4000:
                parts = [message[i:i+4000] for i in range(0, len(message), 4000)]
                for part in parts:
                    telegram_bot.send_message(chat_id, part, parse_mode="HTML")
            else:
                telegram_bot.send_message(chat_id, message, parse_mode="HTML")
                
            logger.info("✅ Relatório de mentoria enviado com sucesso.")
            return True
        except Exception as e:
            logger.error(f"❌ Erro ao enviar relatório de mentoria: {e}")
            return False

# Instância global
mentorship_reporter = MentorshipReporter()

if __name__ == "__main__":
    # Teste rápido
    import MetaTrader5 as mt5
    if mt5.initialize():
        print("Testando geração de relatório...")
        print(mentorship_reporter.generate_mentorship_message())
        mt5.shutdown()
    else:
        print("Erro ao inicializar MT5")
