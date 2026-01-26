# validation.py (new file as per Priority 4)
from dataclasses import dataclass
from typing import Optional
from enum import Enum
import MetaTrader5 as mt5
import logging
import config
import utils

logger = logging.getLogger(__name__)  # Ou "validation" para um nome específico
class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"

@dataclass
class OrderParams:
    """
    Parâmetros validados de ordem.
    Validação automática em __post_init__.
    """
    symbol: str
    side: OrderSide
    volume: float
    entry_price: float
    sl: float
    tp: float
    
    def __post_init__(self):
        """Validação automática após criação"""
        errors = []
        
        # 1. Volume
        if self.volume <= 0:
            errors.append(f"Volume inválido: {self.volume}")
        
        if self.volume % 100 != 0:
            errors.append(f"Volume deve ser múltiplo de 100: {self.volume}")
        
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
        min_rr = utils.get_dynamic_rr_min()  # Já dinâmico (1.25 RISK_ON, 1.5 RISK_OFF)
        regime = utils.detect_market_regime()  # Integra com utils.py
    
        # 🆕 Novo: Aumentar min_rr em RISK_OFF para 2.0
        if regime == "RISK_OFF":
            min_rr = max(min_rr, 2.0)  # Mais conservador em incertezas
        
        if round(rr, 2) < round(min_rr, 2):
            errors.append(f"R:R baixo ({rr:.2f} < {min_rr:.2f}) | Regime: {regime}")
    
        # 🆕 Novo: Check Volatilidade (SL > 1.5 ATR mínimo)
        atr = utils.get_atr(...)  # Assuma que você adiciona uma chamada para ATR do símbolo
        sl_dist = abs(self.entry_price - self.sl)
        if sl_dist < 1.5 * atr:
            errors.append(f"SL muito apertado (<1.5 ATR: {sl_dist:.2f} < {1.5 * atr:.2f})")
    
        if errors:
            logger.error(f"❌ Ordem rejeitada para {self.symbol}: {errors}")  # Novo: Log rejeições
            raise ValueError(...)

    @property
    def risk_reward_ratio(self) -> float:
        """Calcula R:R da ordem"""
        risk = abs(self.entry_price - self.sl)
        reward = abs(self.tp - self.entry_price)
        return reward / risk if risk > 0 else 0
    
    def to_mt5_request(self, magic: int = 123456, comment: str = "XP3") -> dict:
        """Converte para formato MT5"""
        order_type = mt5.ORDER_TYPE_BUY if self.side == OrderSide.BUY else mt5.ORDER_TYPE_SELL
        
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
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

def validate_and_create_order(symbol: str, side: str, volume: float, 
                               entry_price: float, sl: float, tp: float) -> Optional[OrderParams]:
    """
    Factory function com validação.
    Retorna OrderParams validado ou None se inválido.
    """
    try:
        return OrderParams(
            symbol=symbol,
            side=OrderSide[side],
            volume=volume,
            entry_price=entry_price,
            sl=sl,
            tp=tp
        )
    except ValueError as e:
        logger.error(f"❌ Validação falhou: {e}")
        return None