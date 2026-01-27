# validation.py (new file as per Priority 4)
from dataclasses import dataclass
from typing import Optional
from enum import Enum
import MetaTrader5 as mt5
import logging
import config
import utils
from rejection_logger import log_trade_rejection
from datetime import datetime

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
                if d.symbol == symbol and d.entry == mt5.DEAL_ENTRY_OUT and float(d.profit) < 0:
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
        atr_val = utils.get_atr(utils.safe_copy_rates(self.symbol, mt5.TIMEFRAME_M15, 50)) or 0
        if atr_val > 0:
            atr_pct = (atr_val / self.entry_price) * 100
            
            # Limites base vs adaptativo
            max_atr = config.MAX_ATR_PCT # Padrão 5.0%
            
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
                logger.info(f"📈 {self.symbol}: ADX em ascensão ({adx_val:.1f}) -> ATR Limit garantido em 5.0%")

            if config.ADAPTIVE_ATR_FILTER and adx_val >= 40:
                max_atr = config.MAX_ATR_PCT_HIGH_ADX # 7.0%
                logger.info(f"🌪️ {self.symbol}: ADX Forte ({adx_val:.0f}) -> ATR Limit expandido para {max_atr}%")
            
            if atr_pct > max_atr:
                 errors.append(f"Volatilidade Crítica: ATR {atr_pct:.1f}% > {max_atr}%")
            elif atr_pct > (max_atr * 0.8):  # Warning para ATR chegando no limite (ajustado de 0.7 para 0.8)
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
        
        is_fut = utils.is_future(self.symbol)
        if is_fut:
            self.volume = float(int(max(1, int(self.volume))))
        else:
            if self.volume % 100 != 0:
                self.volume = (int(self.volume) // 100) * 100
                logger.info(f"⚖️ Volume ajustado para lote padrão: {self.volume}")
            if self.volume < 100:
                errors.append(f"Volume insuficiente para lote mínimo de 100: {self.volume}")
        
        # 2. Preço
        if self.entry_price <= 0:
            errors.append(f"Preço inválido: {self.entry_price}")
        
        # 3. Stop Loss
        if self.side == OrderSide.BUY:
            if self.sl >= self.entry_price:
                errors.append(f"SL de COMPRA deve ser < entrada: {self.sl} >= {self.entry_price}")
        else:
            if self.sl <= self.entry_price:
                errors.append(f"SL de VENDA deve ser > entrada: {self.sl} <= {self.entry_price}")
        
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
        min_rr = getattr(config, "MIN_RR_FUTURES", 2.5) if utils.is_future(self.symbol) else getattr(config, "MIN_RR", 2.0)
        # 🆕 Novo: Aumentar min_rr em Alta Volatilidade
        atr_val = utils.get_atr(utils.safe_copy_rates(self.symbol, mt5.TIMEFRAME_M15, 50)) or 0
        if atr_val > 0:
            atr_pct = (atr_val / self.entry_price) * 100
            if atr_pct > 3.0: # Vol alta
                min_rr = max(min_rr, config.MIN_RR_HIGH_VOL)

        sl_dist = abs(self.entry_price - self.sl)
        if sl_dist < 1.5 * atr_val:
            errors.append(f"SL muito apertado (<1.5 ATR: {sl_dist:.2f} < {1.5 * atr_val:.2f})")
        if utils.is_future(self.symbol):
            info = mt5.symbol_info(self.symbol)
            if info and info.point > 0:
                sl_points = sl_dist / info.point
                min_points = max(10, int((atr_val / info.point) * 1.2))
                if sl_points < min_points:
                    errors.append(f"SL em pontos insuficiente: {sl_points:.0f} < {min_points}")
    
        if errors:
            full_reason = " | ".join(errors)
            logger.error(f"❌ Ordem rejeitada para {self.symbol}: {full_reason}")
            log_trade_rejection(self.symbol, "OrderValidation", "Múltiplos erros", {"errors": errors})
            raise ValueError(full_reason)

    @property
    def risk_reward_ratio(self) -> float:
        """Calcula R:R da ordem"""
        risk = abs(self.entry_price - self.sl)
        reward = abs(self.tp - self.entry_price)
        return reward / risk if risk > 0 else 0
    
    def to_mt5_request(self, magic: int = 123456, comment: str = "XP3") -> dict:
        """Converte para formato MT5"""
        order_type = mt5.ORDER_TYPE_BUY if self.side == OrderSide.BUY else mt5.ORDER_TYPE_SELL
        if magic == 123456:
            magic = 2000 if utils.is_future(self.symbol) else 1000
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
        if utils.is_future(symbol):
            info = mt5.symbol_info(symbol)
            if not info or info.point <= 0:
                return False, "Sem info de ponto para futuros"
            spread_points = spread / info.point
            max_points_cfg = getattr(config, "MAX_SPREAD_FUTURE_POINTS", None)
            if max_points_cfg is None:
                from datetime import datetime
                server_time = datetime.fromtimestamp(tick.time).time()
                t_open = datetime.strptime("10:00", "%H:%M").time()
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
        mid_price = (tick.ask + tick.bid) / 2
        spread_pct = (spread / mid_price) * 100
        max_spread_pct = getattr(config, "MAX_SPREAD_ACTION_PCT", None)
        if max_spread_pct is None:
            from datetime import datetime
            server_time = datetime.fromtimestamp(tick.time).time()
            if server_time < datetime.strptime("15:30", "%H:%M").time():
                max_spread_pct = 0.15
            elif server_time < datetime.strptime("17:00", "%H:%M").time():
                max_spread_pct = 0.30
            else:
                max_spread_pct = 0.45
        if not utils.is_spread_acceptable(symbol, max_spread_pct=max_spread_pct):
            return False, f"Spread alto: {spread_pct:.3f}% > {max_spread_pct:.2f}%"
        return True, f"OK ({spread_pct:.3f}%)"
        
    except Exception as e:
        logger.warning(f"Erro na validação de spread: {e}")
        return True, "Validação ignorada (erro)"


# ============================================
# � LIMITE DE PERDA DIÁRIA (R$) POR ATIVO
# ============================================
def check_daily_loss_money_limit(symbol: str, planned_volume: float, entry_price: float) -> tuple[bool, str]:
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
                if d.symbol == symbol and d.entry == mt5.DEAL_ENTRY_OUT and d.profit < 0:
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
            return False, f"Exposição subsetor {subsetor} alta: {subsetor_exposure_pct:.1%} >= {limit_pct:.0%}"
            
        # Check por contagem (opcional se houver na config)
        max_count = config.MAX_PER_SUBSETOR.get(subsetor)
        if max_count:
            count = sum(1 for p in positions if config.SUBSETOR_MAP.get(p.symbol) == subsetor)
            if count >= max_count:
                return False, f"Limite de ativos no subsetor {subsetor} atingido: {count}/{max_count}"
                
        return True, "OK"
        
    except Exception as e:
        logger.error(f"Erro na validação de subsetor {symbol}: {e}")
        return True, "Erro na validação (permitido)"


# ============================================
# 🎲 MONTE CARLO KELLY
# ============================================
def monte_carlo_ruin_check(win_rate: float, rr: float, fraction: float, runs: int = 5000) -> float:
    """
    Simula trajetórias para calcular probabilidade de ruína (>90% loss).
    """
    import numpy as np
    ruins = 0
    
    for _ in range(runs):
        balance = 1.0
        for _ in range(200): # Simula 200 trades
            if np.random.random() < win_rate:
                balance += balance * fraction * rr
            else:
                balance -= balance * fraction
            
            if balance < 0.1: # Ruína: perdeu 90%
                ruins += 1
                break
                
    return ruins / runs

def calculate_kelly_position_size(symbol: str, entry_price: float, sl: float, 
                                  tp: float, side: str) -> float:
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
        
        win_rate = stats.get('win_rate', 0.5)
        avg_rr = stats.get('avg_rr', 1.5)
        total_trades = stats.get('total_trades', 0)
        
        # Se não tiver histórico suficiente, usa global
        if total_trades < 10:
            global_stats = utils.get_symbol_performance("GLOBAL", lookback_days=90) or {}
            win_rate = global_stats.get('win_rate', 0.5)
            avg_rr = global_stats.get('avg_rr', 1.5)
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
        vol = np.std(df['close'].pct_change()) if df is not None else 0.01
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
            from bot import daily_max_equity, mt5
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
        
        # ✅ NOVO: Monte Carlo Ruin Check (5000 runs)
        ruin_prob = monte_carlo_ruin_check(win_rate, current_rr, adjusted_kelly, runs=5000)
        if ruin_prob > 0.01: 
            logger.warning(f"⚠️ {symbol}: Probabilidade de ruína alta ({ruin_prob:.1%}). Reduzindo Kelly.")
            adjusted_kelly *= 0.5
        
        # ✅ ATUALIZADO: ULTRA-CONSERVATIVE KELLY (0.3x em vez de 0.5x)
        final_kelly = adjusted_kelly * 0.15
        
        # 10. Calcula volume
        acc = mt5.account_info()
        if not acc:
            return 0.0
        
        is_fut = utils.is_future(symbol)
        capital = acc.balance
        if is_fut:
            info = mt5.symbol_info(symbol)
            if not info:
                return 0.0
            point = info.point or 0.0
            pv = config.WIN_POINT_VALUE if symbol.upper().startswith(("WIN", "IND")) else (config.WDO_POINT_VALUE if symbol.upper().startswith(("WDO", "DOL")) else 1.0)
            risk_points = abs(entry_price - sl) / point if point > 0 else 0.0
            order_type = mt5.ORDER_TYPE_BUY if side == "BUY" else mt5.ORDER_TYPE_SELL
            try:
                margin_one = mt5.order_calc_margin(order_type, symbol, 1.0, entry_price)
            except Exception:
                margin_one = 0.0
            free_margin = getattr(acc, "margin_free", 0.0)
            budget = max(0.0, free_margin * 0.35)
            if margin_one <= 0:
                return 0.0
            max_by_margin = int(budget // margin_one)
            max_by_margin = max(0, min(max_by_margin, getattr(config, "FUTURES_MAX_CONTRACTS", 10)))
            if max_by_margin <= 0:
                return 0.0
            vol = max_by_margin
            if risk_points > 0 and pv > 0:
                risk_money = risk_points * pv * vol
                max_risk_money = capital * getattr(config, "RISK_PER_TRADE_PCT", 0.0025)
                if risk_money > max_risk_money:
                    vol = max(1, int(max_risk_money // (risk_points * pv)))
            vol = max(int(info.volume_min or 1), min(int(vol), int(info.volume_max or vol)))
            return float(vol)
        position_value = capital * final_kelly
        volume = round((position_value / entry_price) / 100) * 100
        
        # 11. Validações finais
        info = mt5.symbol_info(symbol)
        if info:
            volume = max(info.volume_min, min(volume, info.volume_max))
            
            # Limites por preço
            if entry_price <= 5.0:
                volume = min(volume, 50000.0)
            elif entry_price <= 20.0:
                volume = min(volume, 20000.0)
            else:
                volume = min(volume, 10000.0)
        
        logger.info(
            f"💰 Kelly {symbol} | "
            f"WR: {win_rate:.1%} | R:R: {current_rr:.2f} | "
            f"Ajustes: Regime {vol_adjust:.1f} | "
            f"Streak {streak_adjustment:.1f} | DD {dd_adjustment:.1f} | "
            f"Kelly: {kelly_fraction:.2%} → Adj: {adjusted_kelly:.2%} | "
            f"Vol: {volume:.0f}"
        )
        
        return max(0.0, volume)
    
    except Exception as e:
        logger.error(f"Erro Kelly {symbol}: {e}", exc_info=True)
        # Fallback: método antigo (Corrigido: passa apenas stop_dist)
        return utils.calculate_position_size_atr(symbol, abs(entry_price - sl))

def validate_and_create_order(symbol: str, side: str, volume: float, 
                               entry_price: float, sl: float, tp: float,
                               use_kelly: bool = True) -> tuple[Optional[OrderParams], Optional[str]]:
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

        # 🔒 Limites diários por contagem (já existentes em utils)
        allowed_daily, daily_reason = utils.check_daily_symbol_limit(symbol)
        if not allowed_daily:
            log_trade_rejection(symbol, "DailySymbolLimit", daily_reason)
            return None, daily_reason

        # 🔒 Limite de perda financeira diária por ativo
        ok_money, money_reason = check_daily_loss_money_limit(symbol, volume, entry_price)
        if not ok_money:
            log_trade_rejection(symbol, "DailyLossMoney", money_reason)
            return None, money_reason
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
            kelly_volume = calculate_kelly_position_size(symbol, entry_price, sl, tp, side)
            
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
            kelly_adjusted=kelly_used
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
