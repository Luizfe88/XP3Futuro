# log_analyzer.py - Analisador de Logs de Análise XP3 PRO
"""
📊 ANALISADOR DE LOGS DIÁRIOS - XP3 PRO FOREX
✅ Estatísticas completas por dia/semana/mês
✅ Identifica padrões de rejeição
✅ Performance por par, estratégia e horário
✅ Exporta relatórios em TXT e CSV
"""

import re
import os
from pathlib import Path
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Any
import statistics

class LogAnalyzer:
    """Analisador completo de logs de análise"""
    
    def __init__(self, log_dir: str = "analysis_logs"):
        self.log_dir = Path(log_dir)
        
        if not self.log_dir.exists():
            print(f"⚠️ Pasta {log_dir} não encontrada!")
            self.log_dir.mkdir(exist_ok=True)
            print(f"✅ Pasta criada: {log_dir}")
    
    def list_log_files(self) -> List[Path]:
        """Lista todos os arquivos de log disponíveis"""
        return sorted(self.log_dir.glob("analysis_log_*.txt"))
    
    def parse_log_file(self, filepath: Path) -> List[Dict]:
        """
        Extrai todas as análises de um arquivo de log
        
        Returns:
            Lista de dicionários com cada análise
        """
        analyses = []
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Regex para capturar cada bloco de análise
            pattern = r"""
                🕐\s(?P<time>\d{2}:\d{2}:\d{2})\s\|\s
                (?P<symbol>\w+)\s\|\s
                (?P<status>.+?)\n
                ={80}.*?\n
                📊\sSinal:\s(?P<signal>\w+)\s\|\s
                Estratégia:\s(?P<strategy>\w+)\s\|\s
                Score:\s(?P<score>[\d.]+).*?\n
                📈\sIndicadores:.*?\n
                \s+•\sRSI:\s(?P<rsi>[\d.]+).*?\n
                \s+•\sADX:\s(?P<adx>[\d.]+).*?\n
                \s+•\sSpread:\s(?P<spread>[\d.]+).*?\n
                \s+•\sVolume:\s(?P<volume>[\d.]+).*?\n
                \s+•\sTendência\sEMA:\s(?P<ema_trend>\w+).*?\n
                💬\sMotivo:\s(?P<reason>.+)
            """
            
            matches = re.finditer(pattern, content, re.VERBOSE | re.DOTALL)
            
            for match in matches:
                analysis = {
                    'time': match.group('time'),
                    'symbol': match.group('symbol'),
                    'status': match.group('status').strip(),
                    'signal': match.group('signal'),
                    'strategy': match.group('strategy'),
                    'score': float(match.group('score')),
                    'rsi': float(match.group('rsi')),
                    'adx': float(match.group('adx')),
                    'spread': float(match.group('spread')),
                    'volume': float(match.group('volume')),
                    'ema_trend': match.group('ema_trend'),
                    'reason': match.group('reason').strip(),
                    'executed': 'EXECUTADA' in match.group('status')
                }
                analyses.append(analysis)
        
        except Exception as e:
            print(f"⚠️ Erro ao ler {filepath.name}: {e}")
        
        return analyses
    
    def analyze_single_day(self, date_str: str = None) -> Dict:
        """
        Analisa um dia específico
        
        Args:
            date_str: Data no formato YYYY-MM-DD (None = hoje)
        
        Returns:
            Dicionário com estatísticas do dia
        """
        if date_str is None:
            date_str = datetime.now().strftime("%Y-%m-%d")
        
        filepath = self.log_dir / f"analysis_log_{date_str}.txt"
        
        if not filepath.exists():
            return {"error": f"Arquivo não encontrado: {filepath.name}"}
        
        analyses = self.parse_log_file(filepath)
        
        if not analyses:
            return {"error": "Nenhuma análise encontrada no arquivo"}
        
        # Estatísticas gerais
        total = len(analyses)
        executed = sum(1 for a in analyses if a['executed'])
        rejected = total - executed
        execution_rate = (executed / total * 100) if total > 0 else 0
        
        # Por símbolo
        symbols = Counter(a['symbol'] for a in analyses)
        symbol_executed = Counter(a['symbol'] for a in analyses if a['executed'])
        
        # Por estratégia
        strategies = Counter(a['strategy'] for a in analyses if a['strategy'] != 'N/A')
        strategy_executed = Counter(a['strategy'] for a in analyses if a['executed'] and a['strategy'] != 'N/A')
        
        # Motivos de rejeição
        rejections = Counter(a['reason'] for a in analyses if not a['executed'])
        
        # Horários com mais atividade
        hours = Counter(a['time'][:2] for a in analyses)  # Pega só a hora
        
        # Score médio
        scores = [a['score'] for a in analyses]
        avg_score_all = statistics.mean(scores) if scores else 0
        
        scores_executed = [a['score'] for a in analyses if a['executed']]
        avg_score_executed = statistics.mean(scores_executed) if scores_executed else 0
        
        scores_rejected = [a['score'] for a in analyses if not a['executed']]
        avg_score_rejected = statistics.mean(scores_rejected) if scores_rejected else 0
        
        # Indicadores médios
        avg_rsi = statistics.mean(a['rsi'] for a in analyses)
        avg_adx = statistics.mean(a['adx'] for a in analyses)
        avg_spread = statistics.mean(a['spread'] for a in analyses)
        avg_volume = statistics.mean(a['volume'] for a in analyses)
        
        return {
            "date": date_str,
            "total_analyses": total,
            "executed": executed,
            "rejected": rejected,
            "execution_rate": execution_rate,
            "symbols": symbols,
            "symbol_executed": symbol_executed,
            "strategies": strategies,
            "strategy_executed": strategy_executed,
            "rejection_reasons": rejections,
            "hourly_activity": hours,
            "avg_score_all": avg_score_all,
            "avg_score_executed": avg_score_executed,
            "avg_score_rejected": avg_score_rejected,
            "avg_rsi": avg_rsi,
            "avg_adx": avg_adx,
            "avg_spread": avg_spread,
            "avg_volume": avg_volume,
            "raw_analyses": analyses
        }
    
    def analyze_date_range(self, start_date: str, end_date: str) -> Dict:
        """
        Analisa um período de datas
        
        Args:
            start_date: Data inicial (YYYY-MM-DD)
            end_date: Data final (YYYY-MM-DD)
        
        Returns:
            Estatísticas agregadas do período
        """
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        all_analyses = []
        days_analyzed = 0
        
        current = start
        while current <= end:
            date_str = current.strftime("%Y-%m-%d")
            filepath = self.log_dir / f"analysis_log_{date_str}.txt"
            
            if filepath.exists():
                analyses = self.parse_log_file(filepath)
                all_analyses.extend(analyses)
                days_analyzed += 1
            
            current += timedelta(days=1)
        
        if not all_analyses:
            return {"error": "Nenhuma análise encontrada no período"}
        
        # Estatísticas agregadas
        total = len(all_analyses)
        executed = sum(1 for a in all_analyses if a['executed'])
        rejected = total - executed
        
        return {
            "period": f"{start_date} até {end_date}",
            "days_analyzed": days_analyzed,
            "total_analyses": total,
            "executed": executed,
            "rejected": rejected,
            "execution_rate": (executed / total * 100) if total > 0 else 0,
            "symbols": Counter(a['symbol'] for a in all_analyses),
            "rejection_reasons": Counter(a['reason'] for a in all_analyses if not a['executed']),
            "avg_score_all": statistics.mean(a['score'] for a in all_analyses),
            "avg_score_executed": statistics.mean(a['score'] for a in all_analyses if a['executed']) if executed > 0 else 0,
        }
    
    def generate_report(self, stats: Dict) -> str:
        """Gera relatório formatado em texto"""
        
        if "error" in stats:
            return f"❌ Erro: {stats['error']}"
        
        lines = []
        lines.append("="*80)
        lines.append("📊 RELATÓRIO DE ANÁLISES - XP3 PRO FOREX")
        lines.append("="*80)
        
        if "date" in stats:
            lines.append(f"📅 Data: {stats['date']}")
        elif "period" in stats:
            lines.append(f"📅 Período: {stats['period']}")
            lines.append(f"📆 Dias analisados: {stats['days_analyzed']}")
        
        lines.append("")
        
        # Resumo geral
        lines.append("📈 RESUMO GERAL")
        lines.append("-"*80)
        lines.append(f"Total de Análises: {stats['total_analyses']}")
        lines.append(f"✅ Executadas: {stats['executed']} ({stats['execution_rate']:.1f}%)")
        lines.append(f"❌ Rejeitadas: {stats['rejected']} ({100-stats['execution_rate']:.1f}%)")
        lines.append("")
        
        # Score médio
        lines.append("🎯 SCORE MÉDIO")
        lines.append("-"*80)
        lines.append(f"Geral: {stats['avg_score_all']:.1f}")
        lines.append(f"Executadas: {stats['avg_score_executed']:.1f}")
        if 'avg_score_rejected' in stats:
            lines.append(f"Rejeitadas: {stats['avg_score_rejected']:.1f}")
        lines.append("")
        
        # Top 10 pares mais analisados
        lines.append("🏆 TOP 10 PARES MAIS ANALISADOS")
        lines.append("-"*80)
        for symbol, count in stats['symbols'].most_common(10):
            executed_count = stats.get('symbol_executed', {}).get(symbol, 0)
            exec_rate = (executed_count / count * 100) if count > 0 else 0
            lines.append(f"{symbol:10s} | {count:3d} análises | {executed_count:2d} executadas ({exec_rate:.0f}%)")
        lines.append("")
        
        # Estratégias
        if stats.get('strategies'):
            lines.append("🎲 PERFORMANCE POR ESTRATÉGIA")
            lines.append("-"*80)
            for strategy, count in stats['strategies'].most_common():
                executed_count = stats.get('strategy_executed', {}).get(strategy, 0)
                exec_rate = (executed_count / count * 100) if count > 0 else 0
                lines.append(f"{strategy:10s} | {count:3d} sinais | {executed_count:2d} executados ({exec_rate:.0f}%)")
            lines.append("")
        
        # Top 10 motivos de rejeição
        if stats.get('rejection_reasons'):
            lines.append("🚫 TOP 10 MOTIVOS DE REJEIÇÃO")
            lines.append("-"*80)
            for reason, count in stats['rejection_reasons'].most_common(10):
                pct = (count / stats['rejected'] * 100) if stats['rejected'] > 0 else 0
                lines.append(f"{count:3d}x ({pct:5.1f}%) | {reason}")
            lines.append("")
        
        # Horários mais ativos
        if stats.get('hourly_activity'):
            lines.append("🕐 HORÁRIOS MAIS ATIVOS (TOP 10)")
            lines.append("-"*80)
            for hour, count in stats['hourly_activity'].most_common(10):
                lines.append(f"{hour}h: {count:3d} análises")
            lines.append("")
        
        # Indicadores médios
        if 'avg_rsi' in stats:
            lines.append("📊 INDICADORES MÉDIOS")
            lines.append("-"*80)
            lines.append(f"RSI: {stats['avg_rsi']:.1f}")
            lines.append(f"ADX: {stats['avg_adx']:.1f}")
            lines.append(f"Spread: {stats['avg_spread']:.2f} pips")
            lines.append(f"Volume: {stats['avg_volume']:.2f}x")
            lines.append("")
        
        lines.append("="*80)
        
        return "\n".join(lines)
    
    def export_to_csv(self, stats: Dict, output_file: str):
        """Exporta análises para CSV"""
        
        if "raw_analyses" not in stats:
            print("⚠️ Dados brutos não disponíveis para exportação CSV")
            return
        
        try:
            import csv
            
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                fieldnames = ['time', 'symbol', 'signal', 'strategy', 'score', 
                            'rsi', 'adx', 'spread', 'volume', 'ema_trend', 
                            'executed', 'reason']
                
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for analysis in stats['raw_analyses']:
                    writer.writerow({k: analysis.get(k) for k in fieldnames})
            
            print(f"✅ CSV exportado: {output_file}")
        
        except Exception as e:
            print(f"❌ Erro ao exportar CSV: {e}")


# ===========================
# FUNÇÕES AUXILIARES
# ===========================

def analyze_today():
    """Análise rápida do dia de hoje"""
    analyzer = LogAnalyzer()
    stats = analyzer.analyze_single_day()
    report = analyzer.generate_report(stats)
    print(report)
    
    # Salva relatório
    output = Path("reports") / f"report_{datetime.now().strftime('%Y-%m-%d')}.txt"
    output.parent.mkdir(exist_ok=True)
    
    with open(output, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n📁 Relatório salvo em: {output}")


def analyze_week():
    """Análise da última semana"""
    analyzer = LogAnalyzer()
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    stats = analyzer.analyze_date_range(
        start_date.strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d")
    )
    
    report = analyzer.generate_report(stats)
    print(report)
    
    # Salva relatório
    output = Path("reports") / f"weekly_report_{end_date.strftime('%Y-%m-%d')}.txt"
    output.parent.mkdir(exist_ok=True)
    
    with open(output, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n📁 Relatório salvo em: {output}")


def interactive_menu():
    """Menu interativo para análise de logs"""
    
    analyzer = LogAnalyzer()
    
    while True:
        print("\n" + "="*80)
        print("📊 XP3 PRO FOREX - ANALISADOR DE LOGS")
        print("="*80)
        print("\n1. 📅 Analisar dia específico")
        print("2. 📆 Analisar período (range de datas)")
        print("3. 🔍 Analisar hoje")
        print("4. 📈 Analisar última semana")
        print("5. 📂 Listar arquivos de log disponíveis")
        print("6. 💾 Exportar CSV (último relatório)")
        print("0. ❌ Sair")
        
        choice = input("\n➤ Escolha uma opção: ").strip()
        
        if choice == "1":
            date = input("\n📅 Digite a data (YYYY-MM-DD) ou ENTER para hoje: ").strip()
            if not date:
                date = None
            
            stats = analyzer.analyze_single_day(date)
            report = analyzer.generate_report(stats)
            print("\n" + report)
            
            save = input("\n💾 Salvar relatório? (s/n): ").strip().lower()
            if save == 's':
                filename = f"report_{date or datetime.now().strftime('%Y-%m-%d')}.txt"
                output = Path("reports") / filename
                output.parent.mkdir(exist_ok=True)
                
                with open(output, 'w', encoding='utf-8') as f:
                    f.write(report)
                
                print(f"✅ Salvo em: {output}")
        
        elif choice == "2":
            start = input("\n📅 Data inicial (YYYY-MM-DD): ").strip()
            end = input("📅 Data final (YYYY-MM-DD): ").strip()
            
            if start and end:
                stats = analyzer.analyze_date_range(start, end)
                report = analyzer.generate_report(stats)
                print("\n" + report)
                
                save = input("\n💾 Salvar relatório? (s/n): ").strip().lower()
                if save == 's':
                    filename = f"range_report_{start}_to_{end}.txt"
                    output = Path("reports") / filename
                    output.parent.mkdir(exist_ok=True)
                    
                    with open(output, 'w', encoding='utf-8') as f:
                        f.write(report)
                    
                    print(f"✅ Salvo em: {output}")
        
        elif choice == "3":
            analyze_today()
            input("\n⏎ Pressione ENTER para continuar...")
        
        elif choice == "4":
            analyze_week()
            input("\n⏎ Pressione ENTER para continuar...")
        
        elif choice == "5":
            files = analyzer.list_log_files()
            print("\n📂 ARQUIVOS DE LOG DISPONÍVEIS:")
            print("-"*80)
            
            if files:
                for f in files:
                    size = f.stat().st_size / 1024  # KB
                    print(f"  • {f.name} ({size:.1f} KB)")
            else:
                print("  (Nenhum arquivo encontrado)")
        
        elif choice == "6":
            date = input("\n📅 Data do relatório (YYYY-MM-DD) ou ENTER para hoje: ").strip()
            if not date:
                date = datetime.now().strftime("%Y-%m-%d")
            
            stats = analyzer.analyze_single_day(date)
            
            if "error" not in stats:
                output = Path("exports") / f"analysis_{date}.csv"
                output.parent.mkdir(exist_ok=True)
                analyzer.export_to_csv(stats, str(output))
            else:
                print(f"❌ {stats['error']}")
        
        elif choice == "0":
            print("\n👋 Até logo!\n")
            break
        
        else:
            print("\n⚠️ Opção inválida!")


# ===========================
# MAIN
# ===========================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "today":
            analyze_today()
        
        elif command == "week":
            analyze_week()
        
        elif command == "date" and len(sys.argv) > 2:
            analyzer = LogAnalyzer()
            stats = analyzer.analyze_single_day(sys.argv[2])
            print(analyzer.generate_report(stats))
        
        elif command == "range" and len(sys.argv) > 3:
            analyzer = LogAnalyzer()
            stats = analyzer.analyze_date_range(sys.argv[2], sys.argv[3])
            print(analyzer.generate_report(stats))
        
        else:
            print("Uso:")
            print("  python log_analyzer.py today         # Analisa hoje")
            print("  python log_analyzer.py week          # Analisa última semana")
            print("  python log_analyzer.py date 2026-01-05    # Analisa data específica")
            print("  python log_analyzer.py range 2026-01-01 2026-01-07    # Analisa período")
    else:
        # Modo interativo
        interactive_menu()