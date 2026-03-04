# daily_learning_report.py
"""
📚 RELATÓRIO DIÁRIO DE APRENDIZADO - XP3 PRO
==============================================
Gerado automaticamente ao fim de cada sessão de trading.

Funções:
  1. Analisa trades do dia (xp3_trades.db)
  2. Compara pesos adaptativos de hoje vs. ontem
  3. Extrai "lições" em linguagem legível
  4. Auto-aplica ajustes nos pesos (nudge ±2% baseado em win rate)
  5. Salva relatório .txt em relatorios/
  6. Retorna texto para envio via Telegram
"""

import os
import json
import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger("daily_learning_report")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)

REPORTS_DIR = Path("relatorios")
REPORTS_DIR.mkdir(exist_ok=True)

# Nudge máximo permitido por sessão (evita oscilação excessiva)
MAX_NUDGE = 0.05   # ±5%
BASE_NUDGE = 0.02  # ±2% padrão

# Win rate mínimo para considerar o símbolo "bom"
WR_GOOD_THRESHOLD = 0.55
WR_BAD_THRESHOLD  = 0.40

# Mapeamento de score_key → descrição legível
WEIGHT_LABELS = {
    "ADX_STRONG":   "ADX forte",
    "ADX_GOOD":     "ADX bom",
    "RSI_EXTREME":  "RSI extremo",
    "RSI_STRETCH":  "RSI esticado",
    "VOL_HIGH":     "Volume alto",
    "VOL_OK":       "Volume OK",
    "EMA_TREND":    "Tendência EMA",
    "EMA_COUNTER":  "Contra-tendência EMA",
    "MOMENTUM_POS": "Momentum positivo",
    "MACD_CROSS":   "Cruzamento MACD",
    "ML_BOOST":     "Boost ML",
}


# =============================================================================
# COLETA DE DADOS
# =============================================================================

def _get_today_trades(date_str: str = None, db_path: str = "xp3_trades.db") -> list:
    """Retorna todos os trades fechados de uma data (lista de dicts)."""
    if date_str is None:
        date_str = datetime.now().date().isoformat()
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM trades WHERE date(timestamp) = date(?) AND exit_price IS NOT NULL",
            (date_str,),
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        logger.warning(f"Erro ao ler trades do banco ({date_str}): {e}")
        return []


def _compute_stats(trades: list) -> dict:
    """
    Calcula estatísticas por símbolo e por estratégia a partir de uma lista de trades.
    Retorna:
    {
        "overall": {total, wins, losses, win_rate, total_pnl},
        "by_symbol": { sym: {total, wins, losses, win_rate, total_pnl} },
        "by_strategy": { strat: {total, wins, losses, win_rate} },
    }
    """
    overall = {"total": 0, "wins": 0, "losses": 0, "total_pnl": 0.0}
    by_symbol: dict = {}
    by_strategy: dict = {}

    for t in trades:
        sym   = str(t.get("symbol") or "UNKNOWN")
        strat = str(t.get("strategy") or "ELITE")
        pnl   = float(t.get("pnl_money") or 0.0)
        win   = pnl > 0

        overall["total"]    += 1
        overall["wins"]     += int(win)
        overall["losses"]   += int(not win)
        overall["total_pnl"] += pnl

        if sym not in by_symbol:
            by_symbol[sym] = {"total": 0, "wins": 0, "losses": 0, "total_pnl": 0.0}
        by_symbol[sym]["total"]    += 1
        by_symbol[sym]["wins"]     += int(win)
        by_symbol[sym]["losses"]   += int(not win)
        by_symbol[sym]["total_pnl"] += pnl

        if strat not in by_strategy:
            by_strategy[strat] = {"total": 0, "wins": 0, "losses": 0}
        by_strategy[strat]["total"]  += 1
        by_strategy[strat]["wins"]   += int(win)
        by_strategy[strat]["losses"] += int(not win)

    # Calcula win_rates
    def _wr(d):
        return d["wins"] / d["total"] if d["total"] > 0 else 0.5
    overall["win_rate"] = _wr(overall)
    for d in by_symbol.values():
        d["win_rate"] = _wr(d)
    for d in by_strategy.values():
        d["win_rate"] = _wr(d)

    return {"overall": overall, "by_symbol": by_symbol, "by_strategy": by_strategy}


# =============================================================================
# EXTRAÇÃO DE LIÇÕES
# =============================================================================

def _extract_lessons(stats: dict, sym_w_today: dict, sym_w_yesterday: dict) -> list:
    """
    Retorna lista de dicts com as lições aprendidas:
    [{"symbol": str, "lesson": str, "adjustments": {key: delta_pct}}]
    """
    lessons = []
    by_symbol = stats.get("by_symbol", {})

    for sym, s in by_symbol.items():
        total = s["total"]
        if total < 2:
            # Dados insuficientes — não tira conclusões
            continue

        wr = s["win_rate"]
        pnl = s["total_pnl"]
        adjustments = {}

        if wr >= WR_GOOD_THRESHOLD:
            # Símbolo performou bem → reforca pesos de qualidade (EMA + ADX)
            magnitude = min(MAX_NUDGE, BASE_NUDGE * (wr - WR_GOOD_THRESHOLD + 0.5))
            adjustments = {
                "EMA_TREND":   +magnitude,
                "ADX_STRONG":  +magnitude,
                "MOMENTUM_POS": +magnitude * 0.5,
            }
            lesson = (
                f"{sym}: WR {wr:.0%} ({s['wins']}W/{s['losses']}L) | "
                f"PnL R$ {pnl:+,.2f} → tendência confirmada, reforçando EMA+ADX"
            )
        elif wr <= WR_BAD_THRESHOLD:
            # Símbolo performou mal → atenua pesos que levaram a entradas ruins
            magnitude = min(MAX_NUDGE, BASE_NUDGE * (WR_BAD_THRESHOLD - wr + 0.5))
            adjustments = {
                "ADX_STRONG":   -magnitude,
                "RSI_EXTREME":  -magnitude * 0.5,
                "MOMENTUM_POS": -magnitude,
            }
            lesson = (
                f"{sym}: WR {wr:.0%} ({s['wins']}W/{s['losses']}L) | "
                f"PnL R$ {pnl:+,.2f} → sinal fraco, atenuando filtros agressivos"
            )
        else:
            # Resultado neutro → registra mas sem mudança
            lesson = (
                f"{sym}: WR {wr:.0%} ({s['wins']}W/{s['losses']}L) | "
                f"PnL R$ {pnl:+,.2f} → resultado neutro, pesos mantidos"
            )

        lessons.append({
            "symbol": sym,
            "lesson": lesson,
            "adjustments": adjustments,
        })

    return lessons


# =============================================================================
# AUTO-APLICAÇÃO DAS LIÇÕES
# =============================================================================

def _apply_lessons_to_weights(lessons: list, symbol_weights: dict, sector_weights: dict) -> dict:
    """
    Aplica os ajustes de peso em symbol_weights (in-place) e retorna um dict
    com os ajustes aplicados: {sym: {key: (old, new)}}
    """
    applied = {}
    base_weight_keys = set(WEIGHT_LABELS.keys())

    for lsn in lessons:
        sym = lsn["symbol"]
        adjs = lsn.get("adjustments", {})
        if not adjs:
            continue

        # Garante que o símbolo existe em symbol_weights
        if sym not in symbol_weights:
            symbol_weights[sym] = {k: 1.0 for k in base_weight_keys}

        applied[sym] = {}
        for key, delta in adjs.items():
            if key not in base_weight_keys:
                continue
            old_val = float(symbol_weights[sym].get(key, 1.0))
            new_val = max(0.5, min(1.8, old_val + delta))
            symbol_weights[sym][key] = round(new_val, 4)
            applied[sym][key] = (round(old_val, 4), round(new_val, 4))

    return applied


# =============================================================================
# FORMATAÇÃO DO RELATÓRIO
# =============================================================================

def _format_report(
    date_str: str,
    stats: dict,
    lessons: list,
    applied: dict,
    sym_w_yesterday: dict,
    sym_w_today: dict,
) -> str:
    """Gera o texto completo do relatório diário de aprendizado."""
    ov = stats["overall"]
    lines = []

    lines.append("╔══════════════════════════════════════════════════════╗")
    lines.append(f"║  📚 RELATÓRIO DE APRENDIZADO - {date_str}          ║")
    lines.append("╚══════════════════════════════════════════════════════╝")
    lines.append("")

    # ── Resumo do dia
    lines.append("📊 RESUMO DO DIA")
    lines.append("─" * 40)
    lines.append(f"Trades fechados : {ov['total']}")
    wr_pct = ov['win_rate'] * 100
    wr_emoji = "✅" if ov['win_rate'] >= WR_GOOD_THRESHOLD else ("⚠️" if ov['win_rate'] >= WR_BAD_THRESHOLD else "❌")
    lines.append(f"Win Rate geral  : {wr_emoji} {wr_pct:.1f}% ({ov['wins']}W / {ov['losses']}L)")
    pnl_emoji = "🟢" if ov['total_pnl'] >= 0 else "🔴"
    lines.append(f"PnL total       : {pnl_emoji} R$ {ov['total_pnl']:+,.2f}")
    lines.append("")

    # ── Por estratégia
    by_strat = stats.get("by_strategy", {})
    if by_strat:
        lines.append("🎯 PERFORMANCE POR ESTRATÉGIA")
        lines.append("─" * 40)
        for strat, s in sorted(by_strat.items(), key=lambda x: -x[1]["win_rate"]):
            emoji = "✅" if s["win_rate"] >= WR_GOOD_THRESHOLD else ("⚠️" if s["win_rate"] >= WR_BAD_THRESHOLD else "❌")
            lines.append(f"  {emoji} {strat:15s} WR {s['win_rate']:.0%}  ({s['wins']}W/{s['losses']}L)")
        lines.append("")

    # ── Por símbolo
    by_sym = stats.get("by_symbol", {})
    if by_sym:
        lines.append("🏆 PERFORMANCE POR SÍMBOLO")
        lines.append("─" * 40)
        for sym, s in sorted(by_sym.items(), key=lambda x: -x[1]["win_rate"]):
            emoji = "✅" if s["win_rate"] >= WR_GOOD_THRESHOLD else ("⚠️" if s["win_rate"] >= WR_BAD_THRESHOLD else "❌")
            lines.append(
                f"  {emoji} {sym:10s} WR {s['win_rate']:.0%}  "
                f"({s['wins']}W/{s['losses']}L)  PnL R$ {s['total_pnl']:+,.2f}"
            )
        lines.append("")

    # ── Lições
    if lessons:
        lines.append("🧠 LIÇÕES APRENDIDAS")
        lines.append("─" * 40)
        for i, lsn in enumerate(lessons, 1):
            lines.append(f"  {i}. {lsn['lesson']}")
        lines.append("")
    else:
        lines.append("🧠 LIÇÕES APRENDIDAS")
        lines.append("─" * 40)
        if ov["total"] == 0:
            lines.append("  ℹ️  Nenhum trade fechado hoje — sem lições para extrair.")
        else:
            lines.append("  ℹ️  Todos os símbolos com dados insuficientes (< 2 trades) para lições.")
        lines.append("")

    # ── Ajustes aplicados
    if applied:
        lines.append("🔧 AJUSTES AUTO-APLICADOS (em vigor amanhã)")
        lines.append("─" * 40)
        for sym, keys in applied.items():
            for key, (old, new) in keys.items():
                direction = "↑" if new > old else "↓"
                delta = new - old
                label = WEIGHT_LABELS.get(key, key)
                lines.append(
                    f"  {direction} {sym:10s} {label:20s}: "
                    f"{old:.4f} → {new:.4f}  ({delta:+.4f})"
                )
        lines.append("")
    else:
        lines.append("🔧 AJUSTES AUTO-APLICADOS")
        lines.append("─" * 40)
        lines.append("  ✅ Nenhum ajuste necessário — pesos mantidos.")
        lines.append("")

    # ── Rodapé
    lines.append("─" * 54)
    lines.append(f"  Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    lines.append("  XP3 PRO — Sistema de Aprendizado Adaptativo")
    lines.append("─" * 54)

    return "\n".join(lines)


def _format_telegram(date_str: str, stats: dict, lessons: list, applied: dict) -> str:
    """Versão condensada para Telegram (HTML)."""
    ov = stats["overall"]
    wr_pct = ov["win_rate"] * 100
    wr_emoji = "✅" if ov["win_rate"] >= WR_GOOD_THRESHOLD else ("⚠️" if ov["win_rate"] >= WR_BAD_THRESHOLD else "❌")
    pnl_emoji = "🟢" if ov["total_pnl"] >= 0 else "🔴"

    msg_lines = [
        f"📚 <b>APRENDIZADO DIÁRIO — {date_str}</b>",
        "",
        f"📊 <b>Trades:</b> {ov['total']} | {wr_emoji} WR {wr_pct:.1f}% ({ov['wins']}W/{ov['losses']}L)",
        f"{pnl_emoji} <b>PnL:</b> R$ {ov['total_pnl']:+,.2f}",
        "",
    ]

    if lessons:
        msg_lines.append("<b>🧠 Lições:</b>")
        for lsn in lessons[:5]:  # máx 5 para não estourar Telegram
            sym = lsn["symbol"]
            adjs = lsn.get("adjustments", {})
            if adjs:
                adj_str = ", ".join(f"{WEIGHT_LABELS.get(k,k)} {'+' if v>0 else ''}{v*100:.0f}%" for k, v in adjs.items())
                msg_lines.append(f"  • {sym}: {adj_str}")
            else:
                msg_lines.append(f"  • {lsn['lesson'][:80]}")
        msg_lines.append("")

    n_adj = sum(len(v) for v in applied.values())
    if n_adj > 0:
        msg_lines.append(f"🔧 <b>{n_adj} ajuste(s) aplicado(s)</b> nos pesos adaptativos.")
    else:
        msg_lines.append("✅ Pesos mantidos (sem ajuste necessário).")

    msg_lines.append(f"\n<i>⏱ {datetime.now().strftime('%H:%M:%S')}</i>")
    return "\n".join(msg_lines)


# =============================================================================
# CLASSE PRINCIPAL
# =============================================================================

class DailyLearningReport:
    """
    Gera o relatório diário de aprendizado, extrai lições e auto-aplica
    ajustes nos pesos adaptativos do sistema.
    """

    def __init__(self, db_path: str = "xp3_trades.db"):
        self.db_path = db_path

    def generate_and_apply(self, date_str: str = None) -> str:
        """
        Pipeline completo:
        1. Coleta trades do dia
        2. Calcula estatísticas
        3. Carrega pesos de ontem e de hoje do banco
        4. Extrai lições
        5. Auto-aplica ajustes nos pesos em memória + persiste
        6. Salva relatório .txt
        7. Retorna texto Telegram

        Retorna: string HTML para Telegram (ou string vazia em caso de falha).
        """
        if date_str is None:
            date_str = datetime.now().date().isoformat()

        logger.info(f"📚 Gerando Relatório de Aprendizado para {date_str}...")

        # ── 1. Trades do dia
        trades = _get_today_trades(date_str, self.db_path)
        logger.info(f"   → {len(trades)} trades carregados")

        # ── 2. Estatísticas
        stats = _compute_stats(trades)

        # ── 3. Pesos: hoje e ontem do banco
        try:
            import database as _db
            sym_w_today, sec_w_today = _db.load_adaptive_weights_snapshot(date_str)
            yesterday = (datetime.now().date() - timedelta(days=1)).isoformat()
            sym_w_yesterday, _ = _db.load_adaptive_weights_snapshot(yesterday)
        except Exception as e:
            logger.warning(f"Não foi possível carregar pesos do banco: {e}")
            sym_w_today = sym_w_yesterday = None

        # Usa pesos em memória (utils) se não encontrou no banco
        try:
            import utils as _utils
            live_sym_w = _utils.symbol_weights or {}
            live_sec_w = _utils.sector_weights or {}
        except Exception:
            live_sym_w = {}
            live_sec_w = {}

        sym_w_today     = sym_w_today     or live_sym_w
        sym_w_yesterday = sym_w_yesterday or {}

        # ── 4. Lições
        lessons = _extract_lessons(stats, sym_w_today, sym_w_yesterday)
        logger.info(f"   → {len(lessons)} lições extraídas")

        # ── 5. Auto-aplicação nos pesos em memória
        working_sym_w = dict(sym_w_today)  # cópia para não mutar referência original diretamente
        applied = _apply_lessons_to_weights(lessons, working_sym_w, live_sec_w)

        if applied:
            # Propaga de volta para utils (in-place)
            try:
                import utils as _utils
                for sym, keys in applied.items():
                    if sym not in _utils.symbol_weights:
                        _utils.symbol_weights[sym] = {}
                    for key, (_, new_val) in keys.items():
                        _utils.symbol_weights[sym][key] = new_val
                # Persiste
                _utils.save_adaptive_weights()
                logger.info(f"   → {sum(len(v) for v in applied.values())} ajuste(s) aplicado(s) e salvos")
            except Exception as e:
                logger.error(f"Erro ao aplicar pesos em utils: {e}")
        else:
            logger.info("   → Nenhum ajuste necessário")

        # ── 6. Salva relatório .txt
        full_text = _format_report(
            date_str, stats, lessons, applied, sym_w_yesterday, working_sym_w
        )
        report_path = REPORTS_DIR / f"learning_{date_str}.txt"
        try:
            report_path.write_text(full_text, encoding="utf-8")
            logger.info(f"   → Relatório salvo em: {report_path}")
        except Exception as e:
            logger.error(f"Erro ao salvar relatório: {e}")

        # ── 7. Retorna texto Telegram
        return _format_telegram(date_str, stats, lessons, applied)


# Instância global pronta para uso
daily_learner = DailyLearningReport()


# =============================================================================
# EXECUÇÃO STANDALONE (teste manual)
# =============================================================================

if __name__ == "__main__":
    import sys
    import re

    # Fix Windows console encoding (emojis / UTF-8 chars)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    date_arg = sys.argv[1] if len(sys.argv) > 1 else None
    reporter = DailyLearningReport()
    telegram_msg = reporter.generate_and_apply(date_arg)

    print("\n" + "=" * 60)
    print("MENSAGEM TELEGRAM (preview):")
    print("=" * 60)
    # Remove tags HTML para exibição no console
    clean = re.sub(r"<[^>]+>", "", telegram_msg)
    print(clean)
    print("=" * 60)
    date_display = date_arg or datetime.now().date().isoformat()
    print(f"\nRelatorio salvo em: relatorios/learning_{date_display}.txt")
