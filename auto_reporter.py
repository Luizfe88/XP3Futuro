# auto_reporter.py - Gerador Automático de Relatórios
"""
🤖 GERADOR AUTOMÁTICO DE RELATÓRIOS - XP3 PRO
✅ Gera relatórios diários/semanais/mensais automaticamente
✅ Envia via Telegram
✅ Detecta anomalias e alerta
✅ Salva histórico em PDF (opcional)
"""

import sys
import time
from pathlib import Path
from datetime import datetime, timedelta
from log_analyzer import LogAnalyzer

# Tenta importar utils do bot para Telegram
try:
    from utils_forex import send_telegram_message, send_telegram_alert
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    print("⚠️ Telegram não disponível - relatórios serão apenas salvos em arquivo")


class AutoReporter:
    """Gerador automático de relatórios com detecção de anomalias"""
    
    def __init__(self):
        self.analyzer = LogAnalyzer()
        self.reports_dir = Path("reports")
        self.reports_dir.mkdir(exist_ok=True)
    
    def detect_anomalies(self, stats: dict) -> list:
        """
        Detecta anomalias nos dados e retorna lista de alertas
        """
        anomalies = []
        
        if "error" in stats:
            return [f"❌ Erro ao analisar: {stats['error']}"]
        
        # Taxa de execução muito baixa
        if stats['execution_rate'] < 2:
            anomalies.append(
                f"⚠️ Taxa de execução crítica: {stats['execution_rate']:.1f}% "
                f"(apenas {stats['executed']} de {stats['total_analyses']} análises)"
            )
        
        # Taxa de execução muito alta (possível overtrading)
        elif stats['execution_rate'] > 25:
            anomalies.append(
                f"🚨 Taxa de execução muito alta: {stats['execution_rate']:.1f}% "
                f"({stats['executed']} execuções) - Risco de overtrading!"
            )
        
        # Score médio das executadas muito baixo
        if stats.get('avg_score_executed', 100) < 75:
            anomalies.append(
                f"⚠️ Score médio das executadas baixo: {stats['avg_score_executed']:.1f} "
                f"(esperado >85) - Bot pode estar entrando em sinais ruins"
            )
        
        # Diferença pequena entre score executadas vs rejeitadas
        if stats.get('avg_score_executed') and stats.get('avg_score_rejected'):
            diff = stats['avg_score_executed'] - stats['avg_score_rejected']
            if diff < 10:
                anomalies.append(
                    f"⚠️ Diferença de score pequena: {diff:.1f} pontos "
                    f"(executadas {stats['avg_score_executed']:.1f} vs "
                    f"rejeitadas {stats['avg_score_rejected']:.1f}) - "
                    f"Critérios de seleção podem estar fracos"
                )
        
        # Spread médio alto
        if stats.get('avg_spread', 0) > 3:
            anomalies.append(
                f"💰 Spread médio alto: {stats['avg_spread']:.2f} pips "
                f"(esperado <2.5) - Custos de transação elevados"
            )
        
        # Volume médio baixo
        if stats.get('avg_volume', 1) < 0.9:
            anomalies.append(
                f"📉 Volume médio baixo: {stats['avg_volume']:.2f}x "
                f"(esperado >1.0) - Liquidez reduzida"
            )
        
        # ADX médio muito baixo (mercado lateral)
        if stats.get('avg_adx', 30) < 20:
            anomalies.append(
                f"📊 ADX médio muito baixo: {stats['avg_adx']:.1f} "
                f"(esperado >25) - Mercado em range/lateral"
            )
        
        # Nenhuma execução
        if stats['executed'] == 0 and stats['total_analyses'] > 50:
            anomalies.append(
                f"🚫 ZERO execuções com {stats['total_analyses']} análises! "
                f"Filtros podem estar muito restritivos"
            )
        
        # Top motivo de rejeição representa >40% do total
        if stats.get('rejection_reasons'):
            top_reason, top_count = stats['rejection_reasons'].most_common(1)[0]
            if stats['rejected'] > 0:
                pct = (top_count / stats['rejected']) * 100
                if pct > 40:
                    anomalies.append(
                        f"🎯 Motivo dominante de rejeição ({pct:.0f}%): {top_reason}"
                    )
        
        return anomalies
    
    def generate_telegram_summary(self, stats: dict, anomalies: list) -> str:
        """Gera resumo compacto para Telegram"""
        
        if "error" in stats:
            return f"❌ Erro ao gerar relatório: {stats['error']}"
        
        lines = []
        lines.append("📊 <b>RELATÓRIO DIÁRIO - XP3 PRO</b>")
        lines.append("")
        
        # Data
        if "date" in stats:
            lines.append(f"📅 <b>{stats['date']}</b>")
        elif "period" in stats:
            lines.append(f"📅 <b>{stats['period']}</b>")
        lines.append("")
        
        # Resumo
        lines.append(f"📈 <b>Análises:</b> {stats['total_analyses']}")
        lines.append(f"✅ <b>Executadas:</b> {stats['executed']} ({stats['execution_rate']:.1f}%)")
        lines.append(f"❌ <b>Rejeitadas:</b> {stats['rejected']}")
        lines.append("")
        
        # Score
        lines.append(f"🎯 <b>Score Médio:</b> {stats['avg_score_all']:.1f}")
        if stats.get('avg_score_executed'):
            lines.append(f"   • Executadas: {stats['avg_score_executed']:.1f}")
        if stats.get('avg_score_rejected'):
            lines.append(f"   • Rejeitadas: {stats['avg_score_rejected']:.1f}")
        lines.append("")
        
        # Top 3 pares
        if stats.get('symbols'):
            lines.append("<b>🏆 Top 3 Pares:</b>")
            for symbol, count in list(stats['symbols'].most_common(3)):
                exec_count = stats.get('symbol_executed', {}).get(symbol, 0)
                lines.append(f"   • {symbol}: {count} ({exec_count} exec.)")
            lines.append("")
        
        # Top 3 motivos de rejeição
        if stats.get('rejection_reasons'):
            lines.append("<b>🚫 Top 3 Rejeições:</b>")
            for reason, count in list(stats['rejection_reasons'].most_common(3)):
                lines.append(f"   • {count}x: {reason[:40]}")
            lines.append("")
        
        # Anomalias
        if anomalies:
            lines.append("<b>⚠️ ALERTAS:</b>")
            for anomaly in anomalies[:3]:  # Máximo 3 alertas
                lines.append(f"   {anomaly}")
        
        lines.append("")
        lines.append(f"🕐 {datetime.now().strftime('%H:%M:%S')}")
        
        return "\n".join(lines)
    
    def send_daily_report(self):
        """Gera e envia relatório diário"""
        
        print("📊 Gerando relatório diário...")
        
        # Analisa hoje
        stats = self.analyzer.analyze_single_day()
        
        if "error" in stats:
            print(f"❌ {stats['error']}")
            return False
        
        # Detecta anomalias
        anomalies = self.detect_anomalies(stats)
        
        # Gera relatório completo
        full_report = self.analyzer.generate_report(stats)
        
        # Salva em arquivo
        date_str = stats.get('date', datetime.now().strftime('%Y-%m-%d'))
        report_file = self.reports_dir / f"daily_report_{date_str}.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(full_report)
            
            if anomalies:
                f.write("\n\n")
                f.write("="*80 + "\n")
                f.write("⚠️ ANOMALIAS DETECTADAS\n")
                f.write("="*80 + "\n")
                for anomaly in anomalies:
                    f.write(f"\n{anomaly}")
        
        print(f"✅ Relatório salvo: {report_file}")
        
        # Envia via Telegram se disponível
        if TELEGRAM_AVAILABLE:
            try:
                telegram_summary = self.generate_telegram_summary(stats, anomalies)
                send_telegram_message(telegram_summary)
                print("✅ Relatório enviado via Telegram")
            except Exception as e:
                print(f"⚠️ Erro ao enviar Telegram: {e}")
        
        # Alertas críticos separados
        if anomalies and TELEGRAM_AVAILABLE:
            critical_anomalies = [a for a in anomalies if '🚨' in a or 'ZERO' in a]
            if critical_anomalies:
                try:
                    send_telegram_alert(
                        "🚨 <b>ALERTAS CRÍTICOS DETECTADOS</b>\n\n" + 
                        "\n".join(critical_anomalies),
                        level="ERROR"
                    )
                    print("🚨 Alertas críticos enviados")
                except Exception as e:
                    print(f"⚠️ Erro ao enviar alertas: {e}")
        
        return True
    
    def send_weekly_report(self):
        """Gera e envia relatório semanal"""
        
        print("📊 Gerando relatório semanal...")
        
        # Última semana
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        stats = self.analyzer.analyze_date_range(
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d")
        )
        
        if "error" in stats:
            print(f"❌ {stats['error']}")
            return False
        
        # Gera relatório
        full_report = self.analyzer.generate_report(stats)
        
        # Salva
        report_file = self.reports_dir / f"weekly_report_{end_date.strftime('%Y-%m-%d')}.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(full_report)
        
        print(f"✅ Relatório semanal salvo: {report_file}")
        
        # Telegram
        if TELEGRAM_AVAILABLE:
            try:
                summary = (
                    f"📊 <b>RELATÓRIO SEMANAL - XP3 PRO</b>\n\n"
                    f"📅 {stats['period']}\n"
                    f"📆 Dias analisados: {stats['days_analyzed']}\n\n"
                    f"📈 <b>Total:</b> {stats['total_analyses']} análises\n"
                    f"✅ <b>Executadas:</b> {stats['executed']} ({stats['execution_rate']:.1f}%)\n"
                    f"❌ <b>Rejeitadas:</b> {stats['rejected']}\n\n"
                    f"🎯 <b>Score Médio:</b> {stats['avg_score_all']:.1f}\n\n"
                    f"📁 Relatório completo salvo em:\n"
                    f"<code>{report_file.name}</code>"
                )
                
                send_telegram_message(summary)
                print("✅ Resumo semanal enviado via Telegram")
            except Exception as e:
                print(f"⚠️ Erro ao enviar Telegram: {e}")
        
        return True
    
    def monitor_live(self, check_interval: int = 3600):
        """
        Monitora logs em tempo real e envia alertas
        
        Args:
            check_interval: Intervalo de checagem em segundos (padrão: 1 hora)
        """
        
        print(f"🔍 Modo monitoramento ativado (checagem a cada {check_interval}s)")
        print("   Pressione Ctrl+C para parar\n")
        
        last_analysis_count = 0
        
        try:
            while True:
                stats = self.analyzer.analyze_single_day()
                
                if "error" not in stats:
                    current_count = stats['total_analyses']
                    new_analyses = current_count - last_analysis_count
                    
                    if new_analyses > 0:
                        print(f"📊 {datetime.now().strftime('%H:%M:%S')} - "
                              f"Novas análises: {new_analyses} "
                              f"(Total hoje: {current_count})")
                        
                        # Detecta anomalias
                        anomalies = self.detect_anomalies(stats)
                        
                        if anomalies:
                            print(f"⚠️  Anomalias detectadas:")
                            for anomaly in anomalies:
                                print(f"   {anomaly}")
                            
                            # Envia alertas críticos
                            if TELEGRAM_AVAILABLE:
                                critical = [a for a in anomalies if '🚨' in a]
                                if critical:
                                    try:
                                        send_telegram_alert("\n".join(critical), "WARNING")
                                    except:
                                        pass
                        
                        last_analysis_count = current_count
                
                time.sleep(check_interval)
        
        except KeyboardInterrupt:
            print("\n\n👋 Monitoramento interrompido pelo usuário")


def main():
    """Função principal com diferentes modos de operação"""
    
    reporter = AutoReporter()
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == "daily":
            reporter.send_daily_report()
        
        elif mode == "weekly":
            reporter.send_weekly_report()
        
        elif mode == "monitor":
            # Intervalo customizado (padrão 1 hora)
            interval = int(sys.argv[2]) if len(sys.argv) > 2 else 3600
            reporter.monitor_live(interval)
        
        else:
            print("Uso:")
            print("  python auto_reporter.py daily          # Relatório diário")
            print("  python auto_reporter.py weekly         # Relatório semanal")
            print("  python auto_reporter.py monitor [seg]  # Monitoramento contínuo")
    
    else:
        # Modo interativo
        print("\n" + "="*80)
        print("🤖 GERADOR AUTOMÁTICO DE RELATÓRIOS - XP3 PRO")
        print("="*80)
        print("\n1. 📅 Gerar relatório diário")
        print("2. 📆 Gerar relatório semanal")
        print("3. 🔍 Iniciar monitoramento contínuo")
        print("0. ❌ Sair")
        
        choice = input("\n➤ Escolha: ").strip()
        
        if choice == "1":
            reporter.send_daily_report()
        elif choice == "2":
            reporter.send_weekly_report()
        elif choice == "3":
            interval = input("⏱️  Intervalo de checagem (segundos) [3600]: ").strip()
            interval = int(interval) if interval else 3600
            reporter.monitor_live(interval)


if __name__ == "__main__":
    main()