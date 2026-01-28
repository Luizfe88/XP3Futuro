# daily_analysis_logger.py - Sistema de Log Diário de Análises
"""
📝 Logger de análises de sinais em arquivos TXT diários
✅ Um arquivo por dia (analysis_log_YYYY-MM-DD.txt)
✅ Registra TODAS as análises (executadas e rejeitadas)
✅ Formato legível para auditoria manual
"""

import os
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Optional, Dict, Any, List, Tuple

class DailyAnalysisLogger:
    """
    Logger que cria um arquivo TXT novo para cada dia
    registrando todas as análises de sinais
    """
    
    def __init__(self, log_dir: str = "analysis_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.current_date = None
        self.current_file = None
        self.lock = Lock()
        self.rejections = {} # symbol -> {reason: count, timestamp: last}
        self.strategy_stats = {}  # strategy -> {wins: int, losses: int, ml_confidence_sum: float}
        self.executed_trades = []  # Lista de trades executados para análise
        
    def _get_log_filename(self) -> Path:
        """Retorna o nome do arquivo de log para hoje"""
        today = datetime.now().strftime("%Y-%m-%d")
        return self.log_dir / f"analysis_log_{today}.txt"
    
    def _check_date_rollover(self):
        """Verifica se mudou o dia e cria novo arquivo se necessário"""
        today = datetime.now().date()
        
        if self.current_date != today:
            self.current_date = today
            self.current_file = self._get_log_filename()
            
            # Cria arquivo com cabeçalho se não existir
            if not self.current_file.exists():
                with open(self.current_file, 'a', encoding='utf-8') as f:
                    f.write(f"📊 XP3 PRO FOREX - LOG DE ANÁLISES\n")
                    f.write(f"📅 Data: {today.strftime('%d/%m/%Y')}\n")
                    f.write("="*80 + "\n\n")
            
            # Reset rejections on new day
            self.rejections.clear()
            self.strategy_stats.clear()
            self.executed_trades.clear()
    
    def log_analysis(self, 
                     symbol: str,
                     signal: str,
                     strategy: str,
                     score: float,
                     rejected: bool,
                     reason: str,
                     indicators: dict):
        """
        Registra uma análise no arquivo do dia
        
        Args:
            symbol: Par analisado (ex: EURUSD)
            signal: Sinal detectado (BUY, SELL, NONE)
            strategy: Estratégia usada (TREND, REVERSION, N/A)
            score: Score calculado (0-120)
            rejected: Se foi rejeitado ou executado
            reason: Motivo da rejeição ou execução
            indicators: Dict com RSI, ADX, spread, etc
        """
        
        with self.lock:
            try:
                # Verifica se precisa criar novo arquivo
                self._check_date_rollover()
                
                # Formata timestamp
                timestamp = datetime.now().strftime("%H:%M:%S")
                
                # ==========================================
                # ✅ DINAMIZAÇÃO DO MOTIVO (DYNAMIC REASON)
                # ==========================================
                rsi = indicators.get("rsi", 0)
                adx = indicators.get("adx", 0)
                volume_ratio = indicators.get("volume_ratio", 0)
                score_log = indicators.get("score_log", {})
                
                # Regras de prioridade para o "Motivo"
                dynamic_reason = reason # Fallback
                
                if rejected:
                    if rsi > 70:
                        dynamic_reason = "Aguardando correção (RSI Esticado)"
                    elif adx < 20:
                        dynamic_reason = "Sem força de tendência (ADX Baixo)"
                    
                    # Ajuste de Volume Dinâmico no Logger (Sincronizado com Bot)
                    elif volume_ratio < (0.5 if (12 <= datetime.now().hour < 14) else 0.8):
                        dynamic_reason = "Volume institucional insuficiente"
                        
                    elif score < 61:
                        # Explicação detalhada do Score
                        factors = []
                        
                        # Penalidades
                        if score_log.get("PENALTY_NO_TREND"):
                            factors.append("Sem tendência clara (ADX < 15)")
                        if score_log.get("PENALTY_COUNTER_TREND"):
                            factors.append("Contra a tendência principal (EMA)")
                            
                        # Bônus ausentes (mais importantes)
                        if not score_log.get("MACD_CROSS"):
                            factors.append("Falta cruzamento MACD")
                        if not score_log.get("VOL_BOOST"):
                            factors.append("Volume abaixo do ideal")
                        if not score_log.get("MOMENTUM"):
                            factors.append("Falta força de momentum")
                        if not score_log.get("RSI_OK") and not score_log.get("RSI_MODERADO"):
                            factors.append("RSI fora da zona ideal")
                            
                        if factors:
                            # Tenta resumir
                            detail = " e ".join(factors[:2]) # Pega os dois primeiros para não ficar gigante
                            if len(factors) > 2:
                                detail += "..."
                            dynamic_reason = f"Configuração de Risco: {detail}"
                        else:
                            dynamic_reason = f"Score insuficiente para estratégia ({score:.0f})"

                # Define status visual
                spread_pct = indicators.get("spread_pct", 0)
                
                if not rejected:
                    status_emoji = "✅ EXECUTADA"
                    status_line = "="
                elif "já aberta" in reason.lower():
                    status_emoji = "📌 POSIÇÃO JÁ ABERTA"
                    status_line = "-"
                elif "limite" in reason.lower():
                    status_emoji = "🚫 LIMITE ATINGIDO"
                    status_line = "-"
                elif "aguardando" in reason.lower() or "pullback" in reason.lower() or "correção" in dynamic_reason.lower():
                    status_emoji = "⏳ AGUARDANDO SETUP"
                    status_line = "-"
                elif "spread" in reason.lower() or "pedágio" in reason.lower() or spread_pct > 0.1:
                    status_emoji = "⚠️ LIQUIDEZ BAIXA" if spread_pct > 0.1 else "❌ SPREAD LARGO"
                    status_line = "-"
                else:
                    status_emoji = "❌ REJEITADA"
                    status_line = "-"
                
                # Monta a entrada do log
                log_entry = []
                log_entry.append(status_line * 80)
                log_entry.append(f"🕐 {timestamp} | {symbol} | {status_emoji}")
                log_entry.append(status_line * 80)
                
                # Sinal e Estratégia
                signal_display = signal if signal else "NONE"
                strategy_display = strategy if strategy else "N/A"
                
                # Barra de Progresso do Setup (Score)
                display_score = score
                progress_warning = ""
                
                # ✅ NOVO: Pesos e Travas Land Trading
                raw_score = display_score

                if adx < 20:
                    display_score = min(display_score, 40)
                    progress_warning = " (Score reduzido: tendência fraca)"

                log_entry.append(
                    f"📊 Score real: {raw_score:.0f} | Score filtrado: {display_score:.0f}"
                )

                
                if spread_pct > 0.1:
                    display_score = min(display_score, 50)
                    progress_warning = " ⚠️ Spread muito alto para operar"
                
                progress_length = 10
                MAX_SCORE = 120
                filled = int(round((display_score / MAX_SCORE) * progress_length))
                bar = "█" * filled + "░" * (progress_length - filled)
                progress_msg = f"📊 Filtros de Setup: [{bar}] {display_score:.0f}%{progress_warning}"
                
                log_entry.append(f"📊 Sinal: {signal_display} | Estratégia: {strategy_display}")
                log_entry.append(progress_msg)
                
                # Indicadores
                spread_nom = indicators.get("spread_nominal", 0)
                spread_pts = indicators.get("spread_points", 0)
                ema_trend = indicators.get("ema_trend", "N/A")
                is_index = any(x in symbol.upper() for x in ["WIN", "IND", "IBOV", "IFNC"])

                if is_index:
                    spread_txt = f"{spread_pts:.0f} pts"
                else:
                    spread_txt = f"R$ {spread_nom:.4f}"

                log_entry.append(f"📈 Indicadores:")
                log_entry.append(f"   • RSI: {rsi:.1f}")
                log_entry.append(f"   • ADX: {adx:.1f}")
                log_entry.append(f"   • Spread: {spread_txt} ({spread_pct:.3f}%)")
                log_entry.append(f"   • Volume: {volume_ratio:.2f}x")
                log_entry.append(f"   • Tendência EMA: {ema_trend}")
                
                # Motivo
                log_entry.append(f"💬 Motivo: {dynamic_reason}")
                
                # ✅ NOVO: Destaque para Sinal Forte Rejeitado
                if rejected and score >= 61:
                    log_entry.append("-" * 40)
                    log_entry.append(f"⚠️  [ALERTA] Sinal Forte detectado, mas ordem não enviada por: {dynamic_reason}")
                    log_entry.append("-" * 40)
                
                # Track rejection for summary
                if rejected:
                    if symbol not in self.rejections:
                        self.rejections[symbol] = {"reasons": {}, "last_time": timestamp, "score": score}
                    r_info = self.rejections[symbol]
                    r_info["reasons"][dynamic_reason] = r_info["reasons"].get(dynamic_reason, 0) + 1
                    r_info["last_time"] = timestamp
                    r_info["score"] = score

                # Escreve no arquivo e garante nova linha para a próxima entrada
                with open(self.current_file, 'a', encoding='utf-8') as f:
                    f.write('\n'.join(log_entry) + "\n\n")
                    f.flush()
                
            except Exception as e:
                # Não queremos que erro no log quebre o bot
                print(f"⚠️ Erro ao escrever log de análise: {e}")
    
    def log_summary(self, total_analyzed: int, executed: int, rejected: int):
        """
        Adiciona um resumo ao final do arquivo
        """
        with self.lock:
            try:
                self._check_date_rollover()
                
                timestamp = datetime.now().strftime("%H:%M:%S")
                
                summary = [
                    "\n" + "="*80,
                    f"📊 RESUMO PARCIAL - {timestamp}",
                    "="*80,
                    f"Total Analisado: {total_analyzed}",
                    f"Ordens Executadas: {executed}",
                    f"Sinais Rejeitados: {rejected}",
                    f"Taxa de Execução: {(executed/total_analyzed*100) if total_analyzed > 0 else 0:.1f}%",
                    "="*80 + "\n",
                ]
                
                with open(self.current_file, 'a', encoding='utf-8') as f:
                    f.write('\n'.join(summary))
                    
            except Exception as e:
                print(f"⚠️ Erro ao escrever resumo: {e}")

    def get_daily_rejection_summary(self) -> str:
        """
        Gera o relatório diário de ativos não comprados.
        """
        with self.lock:
            if not self.rejections:
                return "Nenhuma rejeição registrada hoje."
            
            summary = ["\n📋 RELATÓRIO DE REJEIÇÕES DIÁRIAS", "="*35]
            
            # Ordena por score (mais promissores primeiro)
            sorted_rejections = sorted(
                self.rejections.items(), 
                key=lambda x: x[1]["score"], 
                reverse=True
            )
            
            for symbol, info in sorted_rejections:
                # Pega o motivo mais frequente
                top_reason = max(info["reasons"].items(), key=lambda x: x[1])[0]
                summary.append(
                    f"• {symbol:6} | {info['last_time']} | Score: {info['score']:3.0f} | {top_reason}"
                )
            
            return "\n".join(summary)

    def log_trade_result(self, symbol: str, strategy: str, pnl: float, 
                         ml_confidence: float = 0.0, ml_prediction: str = ""):
        """
        Registra resultado de um trade para cálculo de win rate por estratégia.
        """
        with self.lock:
            if strategy not in self.strategy_stats:
                self.strategy_stats[strategy] = {
                    "wins": 0, "losses": 0, 
                    "ml_confidence_sum": 0, "ml_correct": 0, "ml_total": 0
                }
            
            stats = self.strategy_stats[strategy]
            
            if pnl > 0:
                stats["wins"] += 1
            else:
                stats["losses"] += 1
            
            # ML accuracy tracking
            if ml_confidence > 0:
                stats["ml_confidence_sum"] += ml_confidence
                stats["ml_total"] += 1
                if (
                    (ml_prediction == "BUY" and pnl > 0) or
                    (ml_prediction == "SELL" and pnl < 0)
                ):
                    stats["ml_correct"] += 1

            
            self.executed_trades.append({
                "symbol": symbol,
                "strategy": strategy,
                "pnl": pnl,
                "ml_confidence": ml_confidence,
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })

    def get_strategy_win_rates(self) -> str:
        """
        Retorna win rate por estratégia.
        """
        with self.lock:
            if not self.strategy_stats:
                return "Nenhum trade registrado hoje."
            
            lines = ["\n📊 WIN RATE POR ESTRATÉGIA", "="*40]
            
            for strategy, stats in self.strategy_stats.items():
                total = stats["wins"] + stats["losses"]
                wr = (stats["wins"] / total * 100) if total > 0 else 0
                
                # ML metrics
                ml_acc = ""
                if stats["ml_total"] > 0:
                    ml_accuracy = stats["ml_correct"] / stats["ml_total"] * 100
                    avg_conf = stats["ml_confidence_sum"] / stats["ml_total"]
                    ml_acc = f" | ML: {ml_accuracy:.0f}% acc ({avg_conf:.2f} conf)"
                
                emoji = "✅" if wr >= 55 else "⚠️" if wr >= 45 else "❌"
                lines.append(f"{emoji} {strategy:15} | WR: {wr:5.1f}% | W:{stats['wins']} L:{stats['losses']}{ml_acc}")
            
            return "\n".join(lines)

    def get_ml_performance_summary(self) -> Dict:
        """
        Retorna métricas de performance do ML para integração.
        """
        with self.lock:
            total_trades = len(self.executed_trades)
            ml_trades = [t for t in self.executed_trades if t["ml_confidence"] > 0]
            
            if not ml_trades:
                return {"ml_enabled": False, "total_trades": total_trades}
            
            avg_conf = sum(t["ml_confidence"] for t in ml_trades) / len(ml_trades)
            ml_wins = sum(1 for t in ml_trades if t["pnl"] > 0)
            ml_wr = (ml_wins / len(ml_trades) * 100) if ml_trades else 0
            
            return {
                "ml_enabled": True,
                "total_trades": total_trades,
                "ml_trades": len(ml_trades),
                "ml_win_rate": ml_wr,
                "ml_avg_confidence": avg_conf,
                "ml_wins": ml_wins,
                "ml_losses": len(ml_trades) - ml_wins
            }

# Instância global para usar em todo o bot
daily_logger = DailyAnalysisLogger()


# ===========================
# INTEGRAÇÃO COM O BOT
# ===========================

def log_signal_analysis_to_file(symbol: str, signal: str, strategy: str, score: float,
                                rejected: bool, reason: str, indicators: dict):
    """
    Função wrapper que pode ser chamada no bot_forex.py
    mantendo compatibilidade com o sistema atual
    """
    daily_logger.log_analysis(
        symbol=symbol,
        signal=signal,
        strategy=strategy,
        score=score,
        rejected=rejected,
        reason=reason,
        indicators=indicators
    )


# ===========================
# EXEMPLO DE USO
# ===========================

if __name__ == "__main__":
    # Testes
    logger = DailyAnalysisLogger()
    
    # Exemplo 1: Ordem executada
    logger.log_analysis(
        symbol="EURUSD",
        signal="BUY",
        strategy="TREND",
        score=95,
        rejected=False,
        reason="✅ ORDEM EXECUTADA!",
        indicators={
            "rsi": 35,
            "adx": 42,
            "spread_points": 2,
            "spread_pct": 0.04,
            "volume_ratio": 1.3,
            "ema_trend": "UP"
        }
    )
    
    # Exemplo 2: Aguardando pullback
    logger.log_analysis(
        symbol="GBPUSD",
        signal="NONE",
        strategy="TREND",
        score=65,
        rejected=True,
        reason="⏳ Aguardando pullback (RSI 58 > 40)",
        indicators={
            "rsi": 58,
            "adx": 35,
            "spread_pips": 2.1,
            "volume_ratio": 1.1,
            "ema_trend": "UP"
        }
    )
    
    # Exemplo 3: Correlação alta
    logger.log_analysis(
        symbol="USDCHF",
        signal="SELL",
        strategy="REVERSION",
        score=72,
        rejected=True,
        reason="🔗 Correlação alta",
        indicators={
            "rsi": 75,
            "adx": 18,
            "spread_pips": 2.8,
            "volume_ratio": 0.9,
            "ema_trend": "DOWN"
        }
    )
    
    # Resumo
    logger.log_summary(total_analyzed=50, executed=3, rejected=47)
    
    print("✅ Arquivo de log criado com sucesso!")
    print(f"📁 Localização: {logger.current_file}")
