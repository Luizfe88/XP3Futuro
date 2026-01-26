import MetaTrader5 as mt5
import logging
import config
import numpy as np
from datetime import datetime, timedelta
from threading import Timer
from typing import Dict, Tuple, Optional
import utils

logger = logging.getLogger("hedging")

# =========================================================
# ML PREDICTIVE HEDGING
# =========================================================

class PredictiveHedger:
    """
    Sistema de hedging preditivo usando ML para antecipar riscos.
    """
    
    def __init__(self):
        self.risk_history = []  # Histórico de scores de risco
        self.hedge_active = False
        self.last_prediction = None
    
    def calculate_risk_score(self) -> Dict:
        """
        Calcula score de risco (0-100) baseado em múltiplos fatores.
        """
        try:
            acc = mt5.account_info()
            if not acc:
                return {"score": 50, "factors": {}}
            
            # Fatores de risco
            max_equity = utils.get_daily_max_equity()
            current_dd = (max_equity - acc.equity) / max_equity if max_equity > 0 else 0
            vix_br = utils.get_vix_br()
            
            # Order flow do IBOV
            ibov_flow = utils.get_order_flow("IBOV", bars=10)
            flow_imbalance = ibov_flow.get("imbalance", 0)
            
            # Volatilidade realizada
            df = utils.safe_copy_rates("IBOV", mt5.TIMEFRAME_M15, 20)
            realized_vol = 0
            if df is not None and len(df) > 5:
                realized_vol = df['close'].pct_change().std() * np.sqrt(96 * 252) * 100
            
            # Portfolio heat
            portfolio_heat = 0
            try:
                from bot import get_portfolio_heat
                portfolio_heat = get_portfolio_heat()
            except:
                pass
            
            # Cálculo do score
            score = 0
            factors = {}
            
            # Drawdown atual (0-30 pontos)
            dd_score = min(30, current_dd * 500)
            score += dd_score
            factors["drawdown"] = dd_score
            
            # VIX Brasil (0-25 pontos)
            vix_score = min(25, max(0, (vix_br - 20) * 1.5))
            score += vix_score
            factors["vix"] = vix_score
            
            # Order flow negativo (0-20 pontos)
            if flow_imbalance < -0.3:
                flow_score = min(20, abs(flow_imbalance) * 30)
                score += flow_score
                factors["order_flow"] = flow_score
            
            # Volatilidade alta (0-15 pontos)
            if realized_vol > 25:
                vol_score = min(15, (realized_vol - 25) * 0.5)
                score += vol_score
                factors["volatility"] = vol_score
            
            # Portfolio heat (0-10 pontos)
            heat_score = portfolio_heat * 10
            score += heat_score
            factors["heat"] = heat_score
            
            self.risk_history.append({"time": datetime.now(), "score": score})
            
            # Mantém histórico de 100 pontos
            if len(self.risk_history) > 100:
                self.risk_history = self.risk_history[-100:]
            
            return {"score": min(100, score), "factors": factors, "vix": vix_br, "dd": current_dd}
            
        except Exception as e:
            logger.error(f"Erro ao calcular risk score: {e}")
            return {"score": 50, "factors": {}}
    
    def predict_risk_trend(self) -> str:
        """
        Prediz tendência do risco baseado em histórico recente.
        Returns: 'INCREASING', 'DECREASING', 'STABLE'
        """
        if len(self.risk_history) < 5:
            return "STABLE"
        
        recent = [r["score"] for r in self.risk_history[-10:]]
        older = [r["score"] for r in self.risk_history[-20:-10]] if len(self.risk_history) >= 20 else recent
        
        avg_recent = np.mean(recent)
        avg_older = np.mean(older)
        
        if avg_recent > avg_older + 10:
            return "INCREASING"
        elif avg_recent < avg_older - 10:
            return "DECREASING"
        return "STABLE"
    
    def should_hedge(self) -> Tuple[bool, str]:
        """
        Decide se deve aplicar hedge baseado em predição ML.
        """
        risk = self.calculate_risk_score()
        trend = self.predict_risk_trend()
        
        score = risk["score"]
        
        # Hedge preventivo se risco alto e aumentando
        if score >= 60 and trend == "INCREASING":
            return True, f"Risco alto ({score:.0f}) e aumentando"
        
        # Hedge imediato se risco crítico
        if score >= 75:
            return True, f"Risco crítico ({score:.0f})"
        
        # Hedge se VIX > 35 (volatilidade extrema)
        if risk.get("vix", 0) > 35:
            return True, f"VIX extremo ({risk['vix']:.0f})"
        
        # Hedge se DD > 4% e tendência ruim
        if risk.get("dd", 0) > 0.04 and trend != "DECREASING":
            return True, f"DD alto ({risk['dd']:.1%}) sem melhora"
        
        return False, f"Risco controlado ({score:.0f})"
    
    def calculate_hedge_size(self, risk_score: float) -> float:
        """
        Calcula tamanho do hedge proporcional ao risco.
        """
        exposure = utils.calculate_total_exposure()
        
        # Hedge proporcional: 20-60% do exposure
        if risk_score >= 80:
            hedge_pct = 0.60
        elif risk_score >= 60:
            hedge_pct = 0.40
        elif risk_score >= 40:
            hedge_pct = 0.25
        else:
            hedge_pct = 0.20
        
        return exposure * hedge_pct


# Instância global
predictive_hedger = PredictiveHedger()


def apply_hedge(symbol=None):
    """
    Aplica hedge com DI futures se DD alto ou vol (ex: compra DI1 para hedge taxa).
    ✅ ATUALIZADO: Integração com ML preditivo.
    """
    if not config.ENABLE_HEDGING:
        return
    
    try:
        # ✅ NOVO: Usa predição ML
        should_hedge, reason = predictive_hedger.should_hedge()
        
        if not should_hedge:
            logger.debug(f"Hedge não necessário: {reason}")
            return
        
        logger.warning(f"🛡️ HEDGE PREDITIVO: {reason}")
        
        acc = mt5.account_info()
        if not acc:
            logger.error("Não foi possível obter info da conta")
            return
        
        # Calcula DD atual
        max_equity = utils.get_daily_max_equity()
        current_dd = (max_equity - acc.equity) / max_equity if max_equity > 0 else 0
        vix_br = utils.get_vix_br()
        
        # Verificar Unwind de Hedges existentes
        check_unwind_trigger(current_dd, vix_br)
        
        # Hedge com DI future
        di_symbol = "DI1F25"
        if not mt5.symbol_select(di_symbol, True):
            logger.warning("DI future não disponível, tentando opções")
            hedge_with_options(symbol or "PETR4")
            return
        
        # Calcula volume hedge baseado em risk score
        risk = predictive_hedger.calculate_risk_score()
        hedge_volume = predictive_hedger.calculate_hedge_size(risk["score"])
        
        if hedge_volume <= 0:
            return
        
        # Calcula custo rollover estimado
        rollover_cost = hedge_volume * 0.001
        logger.info(f"Custo rollover estimado: R${rollover_cost:.2f}/dia")
        
        # Envia ordem compra DI
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": di_symbol,
            "volume": hedge_volume,
            "type": mt5.ORDER_TYPE_BUY,
            "price": mt5.symbol_info_tick(di_symbol).ask,
            "deviation": 20,
            "magic": 123456,
            "comment": f"Hedge ML Risk:{risk['score']:.0f}"
        }
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Hedge falhou: {result.comment}")
        else:
            logger.info(f"✅ Hedge ML aplicado: {di_symbol} volume {hedge_volume}")
            predictive_hedger.hedge_active = True
            # Agenda unwind após 24h
            Timer(24*3600, unwind_hedge, args=(result.order, di_symbol)).start()
    
    except Exception as e:
        logger.error(f"Erro hedging: {e}")


def unwind_hedge(order_id, symbol):
    """Fecha hedge automaticamente."""
    positions = mt5.positions_get(symbol=symbol)
    for pos in positions:
        if "Hedge" in pos.comment:
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "position": pos.ticket,
                "symbol": symbol,
                "volume": pos.volume,
                "type": mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                "price": mt5.symbol_info_tick(symbol).bid if pos.type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).ask,
                "deviation": 20,
                "magic": 123456,
                "comment": "Unwind Hedge"
            }
            result = mt5.order_send(request)
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"✅ Hedge desfeito: {symbol}")
                predictive_hedger.hedge_active = False
            else:
                logger.error(f"Falha ao desfazer hedge: {result.comment}")


def check_unwind_trigger(current_dd: float, vix_br: float):
    """Desfaz hedges se DD < 3% ou VIX < 25."""
    if current_dd < config.HEDGE_UNWIND_DD_THRESHOLD or vix_br < config.HEDGE_UNWIND_VIX_THRESHOLD:
        positions = mt5.positions_get()
        if not positions:
            return
            
        for pos in positions:
            if "Hedge" in pos.comment:
                logger.warning(f"🏁 Trigger Unwind: DD={current_dd:.2%}, VIX={vix_br}. Desfazendo {pos.symbol}")
                unwind_hedge(pos.ticket, pos.symbol)


def hedge_with_options(symbol: str):
    """Hedge com opções PUT reais da B3."""
    try:
        all_symbols = mt5.symbols_get(f"{symbol}*")
        if not all_symbols:
            logger.warning(f"Sem opções encontradas para {symbol}")
            return

        # Filtra PUTs (M-X na B3)
        puts = [s.name for s in all_symbols if len(s.name) >= 7 and s.name[4] in "MNOPQRSTUVWX"]
        
        if not puts:
            logger.warning(f"Nenhuma PUT encontrada para {symbol}")
            return
            
        option_symbol = puts[0]
        
        if not mt5.symbol_select(option_symbol, True):
            return
        
        tick = mt5.symbol_info_tick(option_symbol)
        if not tick or tick.ask <= 0:
            return

        risk = predictive_hedger.calculate_risk_score()
        option_volume = max(100, round(predictive_hedger.calculate_hedge_size(risk["score"]) / 100) * 100)
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": option_symbol,
            "volume": float(option_volume),
            "type": mt5.ORDER_TYPE_BUY,
            "price": tick.ask,
            "magic": 999111,
            "comment": f"Hedge PUT ML:{risk['score']:.0f}"
        }
        result = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"✅ Hedge PUT aplicado: {option_symbol} vol {option_volume}")
            predictive_hedger.hedge_active = True
    except Exception as e:
        logger.error(f"Erro hedging opções: {e}")


def get_hedge_status() -> Dict:
    """Retorna status atual do sistema de hedging."""
    risk = predictive_hedger.calculate_risk_score()
    trend = predictive_hedger.predict_risk_trend()
    should, reason = predictive_hedger.should_hedge()
    
    return {
        "risk_score": risk["score"],
        "risk_factors": risk["factors"],
        "trend": trend,
        "should_hedge": should,
        "reason": reason,
        "hedge_active": predictive_hedger.hedge_active
    }
