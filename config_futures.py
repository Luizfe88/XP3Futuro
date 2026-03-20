# config_futures.py
# Configurações Específicas para Futuros B3 (WIN, WDO, DI1, Commodities)

from datetime import time

# ============================================
# 🔧 ESPECIFICAÇÕES DOS CONTRATOS
# ============================================
FUTURES_CONFIGS = {
 
   "WDO$N": {
    "strategy": "MEAN_REVERSION",
    "active": True,
    "params": {
      "bb_period": 23,
      "bb_std": 2.86,
      "tp_mult": 5.85,
      "sl_atr_multiplier": 2.80,
      "adx_threshold": 20.0,
      "base_slippage": 0.0
    },
    "specs": {
      "tick_size": 0.5,
      "point_value": 10.00,
      "value_per_tick": 5.00,
      "margin": 150.00,
      "margin_stress": 3000.00,
      "min_oi": 100000,
      "fees_roundtrip": 1.10,
      "hours": ["09:00", "18:00"],
      "after_market": ["18:00", "18:30"],
      "expiry_day": "First_Business_Day_Month",
      "slippage_base": {"high": 5.0, "avg": 1.0, "low": 0.5, "after": 10.0},
      "min_tick_volume": 50000,
      "min_atr_pct": 0.1,
      "max_spread_points": 4.0
    },
    "note": "🌟 ATIVO ELITE: Calmar 5.66. Configuração Sniper."
  },


  "WIN$N": {
    "strategy": "MEAN_REVERSION",
    "active": True,
    "params": {
      "bb_period": 20,
      "bb_std": 2.00,
      "tp_mult": 1.50,
      "sl_atr_multiplier": 1.50,
      "adx_threshold": 20.0,
      "enable_shorts": 1
    },
    "specs": {
      "tick_size": 5.0,
      "point_value": 0.20,
      "value_per_tick": 1.00,
      "margin": 120.00,
      "margin_stress": 2500.00,
      "min_oi": 50000,
      "fees_roundtrip": 0.50,
      "hours": ["09:00", "17:55"],
      "after_market": ["18:00", "18:25"],
      "expiry_day": "Wednesday_closest_15th_Even_Month",
      "slippage_base": {"high": 15.0, "avg": 5.0, "low": 0.0, "after": 25.0}
    },
    "note": "⚠️ CORREÇÃO: Setup Híbrido aplicado."
  },

  "WSP$N": {
    "strategy": "MEAN_REVERSION",
    "active": True,
    "params": {
      "bb_period": 20,
      "bb_std": 2.00,
      "tp_mult": 1.50,
      "sl_atr_multiplier": 1.50,
      "adx_threshold": 20.0,
      "enable_shorts": 1
    },
    "specs": {
      "tick_size": 0.25,
      "point_value": 2.50,
      "value_per_tick": 0.625,
      "margin": 150.00,
      "margin_stress": 3000.00,
      "min_oi": 10000,
      "fees_roundtrip": 0.80,
      "hours": ["09:00", "17:55"],
      "expiry_day": "Third_Friday_Quarterly",
      "slippage_base": {"high": 2.0, "avg": 0.5, "low": 0.0, "after": 5.0}
    },
    "note": "🆕 NOVO ATIVO: Micro S&P 500. Menor exposição nominal que o WIN."
  },


}

# ============================================
#  fallback
# ============================================
FALLBACK_SYMBOLS = ["WIN$N", "WDO$N"]

# ============================================
# 🕒 HORÁRIOS DE PREGÃO
# ============================================
TRADING_HOURS_FILTER = True

# ============================================
# 💰 GESTÃO DE RISCO E MARGEM
# ============================================
CAPITAL_TOTAL_BASE = 100000.0
MAX_RISK_PERCENT = 0.02
MARGIN_SAFETY_FACTOR = 0.70 # Max 70% do capital em margem stress

