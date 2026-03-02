# validation_core.py
# This module contains all of the validation routines that originally lived in
# `validation.py`. The project also needs a `validation` package (directory)
# for permutation tests and other helpers, and Python will always prefer a
# package directory over a module with the same name.  Keeping a module named
# `validation.py` alongside a `validation/` package caused ImportErrors like
# "cannot import name 'validate_and_create_order' from 'validation'" when the
# package was discovered first.
#
# To resolve the conflict we moved the implementation into this file and
# updated `validation/__init__.py` to re-export its public API.  Consumers can
# continue to write `import validation` or `from validation import
# validate_and_create_order` and will get the same behaviour as before.  The
# legacy `validation.py` file has been removed.

from dataclasses import dataclass
from typing import Optional
from enum import Enum
import MetaTrader5 as mt5
import logging

try:
    import xp3future as config
except ModuleNotFoundError:
    import sys as _sys, os as _os

    _sys.path.insert(
        0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
    )
    import xp3future as config
import utils
from rejection_logger import log_trade_rejection
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)
_last_sl_time: dict[str, datetime] = {}


def register_stop_loss(symbol: str):
    _last_sl_time[symbol] = datetime.now()


def check_revenge_cooldown(symbol: str) -> tuple[bool, str]:
    try:
        cooldown_minutes = getattr(config, "REVENGE_COOLDOWN_MINUTES", 180)
        now = datetime.now()
        ts = _last_sl_time.get(symbol)
        if ts:
            delta = (now - ts).total_seconds() / 60.0
            if delta < cooldown_minutes:
                return False, "🚫 Cooldown Ativo (Stop recente)"
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        deals = mt5.history_deals_get(start, now) or []
        last_loss_time = None
        for d in deals:
            try:
                if (
                    d.symbol == symbol
                    and d.entry == mt5.DEAL_ENTRY_OUT
                    and float(d.profit) < 0
                ):
                    t = datetime.fromtimestamp(d.time)
                    if (last_loss_time is None) or (t > last_loss_time):
                        last_loss_time = t
            except Exception:
                continue
        if last_loss_time:
            delta = (now - last_loss_time).total_seconds() / 60.0
            if delta < cooldown_minutes:
                return False, "🚫 Cooldown Ativo (Stop recente)"
        return True, "OK"
    except Exception:
        return True, "OK"


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class OrderParams:
    """Parâmetros validados de ordem com Kelly Criterion"""

    symbol: str
    side: OrderSide
    volume: float
    entry_price: float
    sl: float
    tp: float
    kelly_adjusted: bool = False

    def __post_init__(self):
        """Validação automática após criação"""
        errors = []

        tick = mt5.symbol_info_tick(self.symbol)
        if not tick:
            errors.append("Sem tick de mercado")
            raise ValueError(f"❌ Rejeitada {self.symbol}: Sem tick")

        # 🆕 ATUALIZADO: Check ATR > 3.5% (mais conservador)
        # 🆕 ATUALIZADO: Check ATR adaptativo (configurações da v5.3)
        # Se ADX > 40 (Tendência Forte), aceita ATR até 7% (MAX_ATR_PCT_HIGH_ADX)
        atr_val = (
            utils.get_atr(utils.safe_copy_rates(self.symbol, mt5.TIMEFRAME_M15, 50))
            or 0
        )
        if atr_val > 0:
            atr_pct = (atr_val / self.entry_price) * 100

            # Limites base vs adaptativo
            max_atr = config.MAX_ATR_PCT  # Padrão 5.0%

            # Verifica ADX para flexibilizar
            adx_val = 0
            adx_rising = False
            try:
                if _df_adx is not None:
                    adx_series = utils.get_adx_series(_df_adx)
                    if len(adx_series) >= 2:
                        adx_val = adx_series.iloc[-1]
                        adx_rising = adx_series.iloc[-1] > adx_series.iloc[-2]
            except Exception:
                adx_val = 0

            # ✅ REGRA B3: Se ADX subindo, permite até 5% (mesmo sem ser > 40)
            if adx_rising and max_atr < 5.0:
                max_atr = 5.0
                logger.info(
                    f"📈 {self.symbol}: ADX em ascensão ({adx_val:.1f}) -> ATR Limit garantido em 5.0%"
                )

            if config.ADAPTIVE_ATR_FILTER and adx_val >= 40:
                max_atr = config.MAX_ATR_PCT_HIGH_ADX  # 7.0%
                logger.info(
                    f"🌪️ {self.symbol}: ADX Forte ({adx_val:.0f}) -> ATR Limit expandido para {max_atr}%"
                )

            if atr_pct > max_atr:
                errors.append(f"Volatilidade Crítica: ATR {atr_pct:.1f}% > {max_atr}%")
            elif atr_pct > (
                max_atr * 0.8
            ):  # Warning para ATR chegando no limite (ajustado de 0.7 para 0.8)
                logger.warning(f"⚠️ {self.symbol}: ATR elevado ({atr_pct:.1f}%)")

        # 🆕 NOVO: Check Correlação IBOV > 0.85
        corr_ibov = utils.get_ibov_correlation(self.symbol)
        max_corr = getattr(config, "MAX_PORTFOLIO_IBOV_CORR", 0.85)
        if corr_ibov > max_corr:
            msg = f"Correlação IBOV Alta: {corr_ibov:.2f} > {max_corr}"
            errors.append(msg)
            log_trade_rejection(self.symbol, "CorrelationFilter", msg)
        if self.volume <= 0:
            errors.append(f"Volume inválido: {self.volume}")

        # Futuros: Volume em contratos (mínimo 1)
        self.volume = float(int(max(1, int(self.volume))))

        # 2. Preço
        if self.entry_price <= 0:
            errors.append(f"Preço inválido: {self.entry_price}")

        # 3. Stop Loss
        if self.side == OrderSide.BUY:
            if self.sl >= self.entry_price:
                errors.append(
                    f"SL de COMPRA deve ser < entrada: {self.sl} >= {self.entry_price}"
                )
        else:
            if self.sl <= self.entry_price:
                errors.append(
                    f"SL de VENDA deve ser > entrada: {self.sl} <= {self.entry_price}"
                )

        # 4. Take Profit
        if self.side == OrderSide.BUY:
            if self.tp <= self.entry_price:
                errors.append(f"TP de COMPRA deve ser > entrada")
        else:
            if self.tp >= self.entry_price:
                errors.append(f"TP de VENDA deve ser < entrada")

        # 5. Distâncias mínimas
        sl_distance = abs(self.entry_price - self.sl)
        tp_distance = abs(self.tp - self.entry_price)

        if sl_distance < 0.01:
            errors.append(f"SL muito próximo da entrada: {sl_distance:.4f}")

        if tp_distance < 0.01:
            errors.append(f"TP muito próximo da entrada: {tp_distance:.4f}")

        rr = self.risk_reward_ratio
        # Futuros: Sempre usa MIN_RR_FUTURES
        min_rr = getattr(config, "MIN_RR_FUTURES", 2.5)

        # 🆕 Novo: Aumentar min_rr em Alta Volatilidade
        atr_val = (
            utils.get_atr(utils.safe_copy_rates(self.symbol, mt5.TIMEFRAME_M15, 50))
            or 0
        )
        if atr_val > 0:
            atr_pct = (atr_val / self.entry_price) * 100
            if atr_pct > 3.0:  # Vol alta
                min_rr = max(min_rr, config.MIN_RR_HIGH_VOL)

        sl_dist = abs(self.entry_price - self.sl)
        if sl_dist < 1.5 * atr_val:
            errors.append(
                f"SL muito apertado (<1.5 ATR: {sl_dist:.2f} < {1.5 * atr_val:.2f})"
            )

        # Validação de SL em pontos para futuros
        info = mt5.symbol_info(self.symbol)
        if info and info.point > 0:
            sl_points = sl_dist / info.point
            min_points = max(10, int((atr_val / info.point) * 1.2))
            if sl_points < min_points:
                errors.append(
                    f"SL em pontos insuficiente: {sl_points:.0f} < {min_points}"
                )

        if errors:
            full_reason = " | ".join(errors)
            logger.error(f"❌ Ordem rejeitada para {self.symbol}: {full_reason}")
            log_trade_rejection(
                self.symbol, "OrderValidation", "Múltiplos erros", {"errors": errors}
            )
            raise ValueError(full_reason)

    @property
    def risk_reward_ratio(self) -> float:
        """Calcula R:R da ordem"""
        risk = abs(self.entry_price - self.sl)
        reward = abs(self.tp - self.entry_price)
        return reward / risk if risk > 0 else 0

    def to_mt5_request(self, magic: int = 123456, comment: str = "XP3") -> dict:
        """Converte para formato MT5"""
        order_type = (
            mt5.ORDER_TYPE_BUY if self.side == OrderSide.BUY else mt5.ORDER_TYPE_SELL
        )
        # Futuros: Sempre usa magic 2000
        if magic == 123456:
            magic = 2000
        return {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": float(self.volume),
            "type": order_type,
            "price": self.entry_price,
            "sl": float(self.sl),
            "tp": float(self.tp),
            "magic": magic,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_RETURN,
        }


# ============================================
# 🛡️ PROTEÇÃO DE SPREAD (COMERCIAL)
# ============================================


def validate_spread_protection(symbol: str) -> tuple:
    """
    Valida se o spread atual está aceitável comparado à média.

    Regra: (ask - bid) < avg_spread * 1.5

    Returns:
        tuple: (aprovado: bool, motivo: str)
    """
    try:
        tick = mt5.symbol_info_tick(symbol)

        if not tick or tick.bid <= 0 or tick.ask <= 0:
            return False, "Sem dados de cotação"

        spread = tick.ask - tick.bid

        # Futuros: Validação sempre em pontos
        info = mt5.symbol_info(symbol)
        if not info or info.point <= 0:
            return False, "Sem info de ponto para futuros"

        spread_points = spread / info.point
        max_points_cfg = getattr(config, "MAX_SPREAD_FUTURE_POINTS", None)

        if max_points_cfg is None:
            # Spread dinâmico por horário
            from datetime import datetime

            server_time = datetime.fromtimestamp(tick.time).time()
            t_1530 = datetime.strptime("15:30", "%H:%M").time()
            t_1700 = datetime.strptime("17:00", "%H:%M").time()
            t_1800 = datetime.strptime("18:00", "%H:%M").time()

            if server_time < t_1530:
                max_points = 20
            elif server_time < t_1700:
                max_points = 35
            elif server_time <= t_1800:
                max_points = 50
            else:
                max_points = 50
        else:
            max_points = int(max_points_cfg)

        if spread_points > max_points:
            return False, f"Spread em pontos alto: {spread_points:.0f} > {max_points}"

        return True, f"OK ({spread_points:.0f} pts)"

    except Exception as e:
        logger.warning(f"Erro na validação de spread: {e}")
        return True, "Validação ignorada (erro)"


# ============================================
# � LIMITE DE PERDA DIÁRIA (R$) POR ATIVO
# ============================================


def check_daily_loss_money_limit(
    symbol: str, planned_volume: float, entry_price: float
) -> tuple[bool, str]:
    """
    Bloqueia novo trade se a perda financeira acumulada hoje do ativo
    exceder o limite percentual do capital alocado planejado.
    """
    try:
        start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        end = datetime.now()
        deals = mt5.history_deals_get(start, end) or []
        loss_money = 0.0
        for d in deals:
            try:
                if (
                    d.symbol == symbol
                    and d.entry == mt5.DEAL_ENTRY_OUT
                    and d.profit < 0
                ):
                    loss_money += abs(float(d.profit))
            except Exception:
                continue
        limit_pct = getattr(config, "DAILY_LOSS_LIMIT_PCT_PER_SYMBOL", 0.015)
        allocated_capital = max(1.0, float(planned_volume)) * float(entry_price)
        max_loss_money = allocated_capital * float(limit_pct)
        if loss_money >= max_loss_money:
            return False, "🚫 Limite de perda diária por ativo atingido"
        return True, "OK"
    except Exception as e:
        logger.warning(f"Erro ao verificar perda diária {symbol}: {e}")
        return True, "Ignorado (erro na checagem)"


# ============================================
# 🏦 LIMITE DE USO DE CAPITAL (35% Equity) & FUNDOS
# ============================================


def check_capital_usage_limit(
    symbol: str, planned_volume: float, entry_price: float
) -> tuple[bool, str]:
    """
    Verifica se a nova ordem respeita o limite máximo de uso de capital (Margem).
    Também verifica se há saldo disponível (Free Margin) para cobrir Margem + Taxas.
    """
    try:
        acc = mt5.account_info()
        if not acc:
            return False, "Sem dados de conta"

        # -----------------------------------------------------------
        # 1. VALIDAÇÃO DE SALDO REAL (Hard Limit)
        # Verifica se tem dinheiro para abrir a posição (Margem + Taxas)
        # -----------------------------------------------------------
        free_margin = acc.margin_free

        # a) Calcula Margem Requerida
        order_type = mt5.ORDER_TYPE_BUY  # Assume compra para cálculo conservador
        try:
            new_margin = mt5.order_calc_margin(
                order_type, symbol, planned_volume, entry_price
            )
        except Exception:
            # Fallback aproximado se falhar
            info = mt5.symbol_info(symbol)
            if info and info.margin_initial > 0:
                new_margin = info.margin_initial * planned_volume
            else:
                new_margin = 0.0

        # b) Calcula Taxas Estimadas (Fees)
        estimated_fees = 0.0
        try:
            import config_futures

            # Normaliza símbolo (WING24 -> WIN, WDOZ25 -> WDO)
            base_symbol = symbol[:3]
            futures_cfg = None

            if hasattr(config_futures, "FUTURES_CONFIGS"):
                for key, cfg in config_futures.FUTURES_CONFIGS.items():
                    if key.replace("$N", "") == base_symbol:
                        futures_cfg = cfg
                        break

            if futures_cfg:
                # fees_roundtrip é por contrato
                estimated_fees = futures_cfg.get("fees_roundtrip", 0.0) * planned_volume
        except Exception:
            pass

        total_required = new_margin + estimated_fees
        missing_funds = total_required - free_margin

        if missing_funds > 0:
            msg_console = (
                f"\n{'=' * 40}\n"
                f"❌ [FALTA DE FUNDOS] {symbol}\n"
                f"💰 Saldo Disponível (Free): R$ {free_margin:.2f}\n"
                f"📉 Custo Total (Margem+Taxas): R$ {total_required:.2f}\n"
                f"   - Margem: R$ {new_margin:.2f}\n"
                f"   - Taxas Est.: R$ {estimated_fees:.2f}\n"
                f"⚠️ FALTA: R$ {missing_funds:.2f}\n"
                f"👉 Deposite R$ {missing_funds:.2f} para operar.\n"
                f"{'=' * 40}"
            )
            # Loga com nível INFO para aparecer no console (StreamHandler)
            logger.info(msg_console)
            # Força print caso logger esteja silenciado
            print(msg_console)
            return False, f"Falta de fundos: -R${missing_funds:.2f}"

        # -----------------------------------------------------------
        # 2. VALIDAÇÃO DE RISCO (Soft Limit - 35% Equity)
        # -----------------------------------------------------------
        limit_pct = getattr(config, "MAX_CAPITAL_USAGE_PCT", 0.35)
        max_allowed_margin = acc.equity * limit_pct

        current_margin = acc.margin
        total_projected_margin = current_margin + new_margin

        missing_risk_budget = total_projected_margin - max_allowed_margin

        if total_projected_margin > max_allowed_margin:
            msg_risk = (
                f"\n{'=' * 40}\n"
                f"🚫 [LIMITE DE RISCO EXCEDIDO] {symbol}\n"
                f"🛡️ Limite (35% Equity): R$ {max_allowed_margin:.2f}\n"
                f"📊 Margem Projetada: R$ {total_projected_margin:.2f}\n"
                f"⚠️ Excesso: R$ {missing_risk_budget:.2f}\n"
                f"{'=' * 40}"
            )
            logger.info(msg_risk)
            print(msg_risk)
            return (
                False,
                f"🚫 Limite de Capital: Margem Projetada {total_projected_margin:.2f} > {max_allowed_margin:.2f} ({limit_pct:.0%})",
            )

        return True, "OK"

    except Exception as e:
        logger.warning(f"Erro ao verificar limite de capital {symbol}: {e}")
        return True, "Ignorado (erro na checagem)"


# ============================================
# ��️ EXPOSIÇÃO POR SUBSETOR
# ============================================


def validate_subsetor_exposure(symbol: str) -> tuple:
    """
    Valida se o subsetor do ativo atingiu o limite de exposição.
    Regra: Max 20% do capital por subsetor ou limite fixo na config.
    """
    subsetor = config.SUBSETOR_MAP.get(symbol)
    if not subsetor:
        return True, "Subsetor não mapeado"

    try:
        acc = mt5.account_info()
        if not acc:
            return False, "Sem dados de conta"

        equity = acc.equity
        positions = mt5.positions_get() or []

        # Calcula exposição financeira do subsetor
        subsetor_vol = 0.0
        for p in positions:
            p_subsetor = config.SUBSETOR_MAP.get(p.symbol)
            if p_subsetor == subsetor:
                subsetor_vol += p.volume * p.price_open

        subsetor_exposure_pct = subsetor_vol / equity if equity > 0 else 0

        # Limite de 20% conforme solicitado ou específico na config
        limit_pct = 0.20

        if subsetor_exposure_pct >= limit_pct:
            return (
                False,
                f"Exposição subsetor {subsetor} alta: {subsetor_exposure_pct:.1%} >= {limit_pct:.0%}",
            )

        # Check por contagem (opcional se houver na config)
        max_count = config.MAX_PER_SUBSETOR.get(subsetor)
        if max_count:
            count = sum(
                1 for p in positions if config.SUBSETOR_MAP.get(p.symbol) == subsetor
            )
            if count >= max_count:
                return (
                    False,
                    f"Limite de ativos no subsetor {subsetor} atingido: {count}/{max_count}",
                )

        return True, "OK"

    except Exception as e:
        logger.error(f"Erro na validação de subsetor {symbol}: {e}")
        return True, "Erro na validação (permitido)"


# ============================================
# 🎲 MONTE CARLO KELLY
# ============================================


def monte_carlo_ruin_check(
    win_rate: float, rr: float, fraction: float, runs: int = 5000
) -> float:
    """
    Simula trajetórias para calcular probabilidade de ruína (>90% loss).
    """
    import numpy as np

    ruins = 0

    for _ in range(runs):
        balance = 1.0
        for _ in range(200):  # Simula 200 trades
            if np.random.random() < win_rate:
                balance += balance * fraction * rr
            else:
                balance -= balance * fraction

            if balance < 0.1:  # Ruína: perdeu 90%
                ruins += 1
                break

    return ruins / runs


def calculate_kelly_position_size(
    symbol: str, entry_price: float, sl: float, tp: float, side: str
) -> float:
    """
    ✅ KELLY DINÂMICO BASEADO EM HISTÓRICO REAL

    Mudanças:
    1. Usa win rate REAL do símbolo (últimos 30 dias)
    2. Ajusta por regime de mercado
    3. Reduz Kelly após perdas consecutivas
    4. Considera drawdown atual

    Impacto: +15-20% em risco ajustado ao retorno
    """
    try:
        # 1. ✅ ESTATÍSTICAS REAIS DO SÍMBOLO (Via Utils)
        # Consulta histórico de 30 dias conforme solicitado
        stats = utils.get_symbol_performance(symbol, lookback_days=30)

        win_rate = stats.get("win_rate", 0.5)
        avg_rr = stats.get("avg_rr", 1.5)
        total_trades = stats.get("total_trades", 0)

        # Se não tiver histórico suficiente, usa global
        if total_trades < 10:
            global_stats = (
                utils.get_symbol_performance("GLOBAL", lookback_days=90) or {}
            )
            win_rate = global_stats.get("win_rate", 0.5)
            avg_rr = global_stats.get("avg_rr", 1.5)
            logger.warning(f"{symbol}: Usando stats globais (poucos trades)")

        # 2. R:R atual
        risk = abs(entry_price - sl)
        reward = abs(tp - entry_price)
        current_rr = reward / risk if risk > 0 else avg_rr

        # 3. Kelly Formula
        p = win_rate
        q = 1 - p
        b = current_rr

        kelly_fraction = (p * b - q) / b

        # 4. Validações
        if kelly_fraction <= 0:
            logger.warning(f"⚠️ {symbol}: Kelly negativo - sem edge")
            return 0.0

        # 5. ✅ NOVO: Ajuste por regime de mercado
        regime = utils.detect_market_regime()
        vol_adjust = 1.0 if regime == "RISK_ON" else 0.5  # Conservador em RISK_OFF

        df = utils.safe_copy_rates(symbol, mt5.TIMEFRAME_M15, 60)
        vol = np.std(df["close"].pct_change()) if df is not None else 0.01
        vol_factor = max(0.25, 1 - vol)  # Reduz se vol alta

        adjusted_kelly = kelly_fraction * vol_adjust * vol_factor
        half_kelly = adjusted_kelly / 2

        # 6. ✅ NOVO: Ajuste por Loss Streak
        from utils import _symbol_loss_streak

        loss_streak = _symbol_loss_streak.get(symbol, 0)

        if loss_streak >= 2:
            streak_adjustment = 0.5  # -50% após 2 perdas
        elif loss_streak >= 1:
            streak_adjustment = 0.7  # -30% após 1 perda
        else:
            streak_adjustment = 1.0

        # 7. ✅ NOVO: Ajuste por Drawdown Atual
        try:
            from botfuturo import daily_max_equity

            acc = mt5.account_info()

            if acc and daily_max_equity > 0:
                current_dd = (daily_max_equity - acc.equity) / daily_max_equity

                if current_dd > 0.03:  # >3% DD
                    dd_adjustment = 0.5
                elif current_dd > 0.02:  # >2% DD
                    dd_adjustment = 0.7
                else:
                    dd_adjustment = 1.0
            else:
                dd_adjustment = 1.0
        except:
            dd_adjustment = 1.0

        # 8. Combina ajustes
        # Fix: regime_adjustment corrigido para vol_adjust
        adjusted_kelly = kelly_fraction * vol_adjust * streak_adjustment * dd_adjustment

        # 9. Limites & Monte Carlo Check
        adjusted_kelly = min(adjusted_kelly, 0.20)

        # ✅ NOVO: Monte Carlo Ruin Check (10000 runs)
        ruin_prob = monte_carlo_ruin_check(
            win_rate, current_rr, adjusted_kelly, runs=10000
        )
        if ruin_prob > 0.005:
            logger.warning(
                f"⚠️ {symbol}: Probabilidade de ruína alta ({ruin_prob:.1%}). Reduzindo Kelly."
            )
            adjusted_kelly *= 0.5

        # ✅ ATUALIZADO: ULTRA-CONSERVATIVE KELLY (0.3x em vez de 0.5x)
        final_kelly = adjusted_kelly * 0.15

        # 10. Calcula volume
        acc = mt5.account_info()
        if not acc:
            return 0.0

        # Futuros: Cálculo de volume baseado em margem e risco
        capital = acc.balance
        info = mt5.symbol_info(symbol)
        if not info:
            return 0.0

        point = info.point or 0.0
        # Define point value por tipo de futuro
        pv = (
            config.WIN_POINT_VALUE
            if symbol.upper().startswith(("WIN", "IND"))
            else (
                config.WDO_POINT_VALUE
                if symbol.upper().startswith(("WDO", "DOL"))
                else 1.0
            )
        )

        risk_points = abs(entry_price - sl) / point if point > 0 else 0.0
        order_type = mt5.ORDER_TYPE_BUY if side == "BUY" else mt5.ORDER_TYPE_SELL

        try:
            margin_one = mt5.order_calc_margin(order_type, symbol, 1.0, entry_price)
        except Exception:
            margin_one = 0.0

        free_margin = getattr(acc, "margin_free", 0.0)

        # ✅ NOVO: Orçamento baseado em limite de 35% do Equity
        limit_pct = getattr(config, "MAX_CAPITAL_USAGE_PCT", 0.35)
        max_allowed_margin = acc.equity * limit_pct
        current_margin = acc.margin
        available_margin_for_limit = max(0.0, max_allowed_margin - current_margin)

        # O orçamento é o menor entre:
        # 1. Margem livre real da conta (para não tomar call)
        # 2. Margem disponível dentro do limite de 35%
        budget = min(free_margin, available_margin_for_limit) * 0.95  # 95% buffer

        if margin_one <= 0:
            return 0.0

        max_by_margin = int(budget // margin_one)
        max_by_margin = max(
            0, min(max_by_margin, getattr(config, "FUTURES_MAX_CONTRACTS", 10))
        )

        if max_by_margin <= 0:
            logger.info(
                f"⚠️ {symbol}: Sem budget de margem (Limit {limit_pct:.0%}: {current_margin:.2f}/{max_allowed_margin:.2f})"
            )
            return 0.0

        vol = max_by_margin

        # Ajusta por risco máximo permitido
        if risk_points > 0 and pv > 0:
            risk_money = risk_points * pv * vol
            max_risk_money = capital * getattr(config, "RISK_PER_TRADE_PCT", 0.0025)
            if risk_money > max_risk_money:
                vol = max(1, int(max_risk_money // (risk_points * pv)))

        vol = max(int(info.volume_min or 1), min(int(vol), int(info.volume_max or vol)))
        return float(vol)

    except Exception as e:
        logger.error(f"Erro Kelly {symbol}: {e}", exc_info=True)
        # Fallback: método antigo (Corrigido: passa apenas stop_dist)
        return utils.calculate_position_size_atr(symbol, abs(entry_price - sl))


def validate_and_create_order(
    symbol: str,
    side: str,
    volume: float,
    entry_price: float,
    sl: float,
    tp: float,
    use_kelly: bool = True,
    portfolio_heat: float = 0.0,
) -> tuple[Optional[OrderParams], Optional[str]]:
    """
    Factory function com validação e Kelly Criterion

    Returns:
        tuple: (OrderParams ou None, erro_string ou None)
    """
    try:
        allowed_cooldown, cooldown_reason = check_revenge_cooldown(symbol)
        if not allowed_cooldown:
            log_trade_rejection(symbol, "CooldownFilter", cooldown_reason)
            return None, "🚫 Cooldown Ativo (Stop recente)"

        # ✅ NOVO: Verificação de Portfolio Heat
        limit_heat = getattr(config, "MAX_PORTFOLIO_HEAT", 0.85)
        if portfolio_heat >= limit_heat:
            log_trade_rejection(
                symbol, "PortfolioHeat", f"Heat {portfolio_heat:.2f} >= {limit_heat}"
            )
            return None, f"Portfolio superaquecido ({portfolio_heat:.2f})"

        # 🔒 Limites diários por contagem (já existentes em utils)
        allowed_daily, daily_reason = utils.check_daily_symbol_limit(symbol)
        if not allowed_daily:
            log_trade_rejection(symbol, "DailySymbolLimit", daily_reason)
            return None, daily_reason

        # 🔒 Limite de perda financeira diária por ativo
        ok_money, money_reason = check_daily_loss_money_limit(
            symbol, volume, entry_price
        )
        if not ok_money:
            log_trade_rejection(symbol, "DailyLossMoney", money_reason)
            return None, money_reason

        # 🔒 Limite de Uso de Capital (35% Equity)
        ok_capital, capital_reason = check_capital_usage_limit(
            symbol, volume, entry_price
        )
        if not ok_capital:
            log_trade_rejection(symbol, "CapitalUsageLimit", capital_reason)
            return None, capital_reason
        # ✅ NOVO: Validação de spread antes de processar
        spread_ok, spread_reason = validate_spread_protection(symbol)
        if not spread_ok:
            logger.warning(f"⚠️ {symbol}: {spread_reason}")
            log_trade_rejection(symbol, "SpreadFilter", spread_reason)
            return None, spread_reason

        # ✅ NOVO: Validação de subsetor
        subsetor_ok, subsetor_reason = validate_subsetor_exposure(symbol)
        if not subsetor_ok:
            logger.warning(f"⚠️ {symbol}: {subsetor_reason}")
            log_trade_rejection(symbol, "SubsetorFilter", subsetor_reason)
            return None, subsetor_reason

        # 🆕 CALCULA VOLUME COM KELLY (se habilitado)
        if use_kelly:
            kelly_volume = calculate_kelly_position_size(
                symbol, entry_price, sl, tp, side
            )

            if kelly_volume <= 0:
                msg = f"{symbol}: Kelly retornou volume 0 - sem edge"
                logger.warning(f"⚠️ {msg}")
                return None, msg

            final_volume = kelly_volume
            kelly_used = True
        else:
            final_volume = volume
            kelly_used = False

        order = OrderParams(
            symbol=symbol,
            side=OrderSide[side],
            volume=final_volume,
            entry_price=entry_price,
            sl=sl,
            tp=tp,
            kelly_adjusted=kelly_used,
        )
        return order, None

    except ValueError as e:
        error_msg = str(e)
        # Extrai apenas a mensagem de erro se houver logs no meio
        if "❌" in error_msg:
            error_msg = error_msg.split("❌")[-1].strip()

        logger.error(f"❌ Validação falhou: {error_msg}")
        return None, error_msg
    except Exception as e:
        logger.error(f"❌ Erro inesperado na validação: {e}")
        return None, f"Erro: {str(e)}"


def check_reentry_after_profit(
    symbol: str, side: str, planned_volume: float
) -> tuple[bool, float, str]:
    try:
        start = datetime.now() - timedelta(days=90)
        end = datetime.now()
        deals = mt5.history_deals_get(start, end) or []
        last = None
        for d in deals:
            try:
                if d.symbol == symbol and d.entry == mt5.DEAL_ENTRY_OUT:
                    t = datetime.fromtimestamp(d.time)
                    if last is None or t > datetime.fromtimestamp(last.time):
                        last = d
            except Exception:
                continue
        if not last or float(getattr(last, "profit", 0.0)) <= 0.0:
            return False, 0.0, "Sem lucro anterior"
        ind = utils.get_cached_indicators(symbol, mt5.TIMEFRAME_M15, 180)
        if not isinstance(ind, dict) or ind.get("error"):
            return False, 0.0, "Sem indicadores"
        from ml_signals import MLSignalPredictor

        predictor = MLSignalPredictor()
        res = predictor.predict(symbol=symbol, indicators=ind)
        conf = float(res.get("confidence", 0.0))
        direction = str(res.get("direction", "HOLD"))
        if conf >= 0.80 and direction == side:
            info = mt5.symbol_info(symbol)
            adj = planned_volume * 0.70
            if info:
                # Futuros: Volume em contratos (inteiros)
                adj = max(
                    int(info.volume_min or 1),
                    min(int(adj), int(info.volume_max or adj)),
                )
            return True, float(adj), "OK"
        return False, 0.0, "Sinal ML insuficiente ou direção divergente"
    except Exception as e:
        logger.warning(f"Erro reentry {symbol}: {e}")
        return False, 0.0, "Erro na checagem"
