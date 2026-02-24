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

  "DOL$N": {
    "strategy": "MEAN_REVERSION",
    "active": True,
    "params": {
      "bb_period": 23,
      "bb_std": 2.86,
      "tp_mult": 5.85,
      "sl_atr_multiplier": 2.80,
      "adx_threshold": 20.0
    },
    "specs": {
      "tick_size": 0.50,
      "point_value": 50.00,
      "value_per_tick": 25.00,
      "margin": 750.00,
      "margin_stress": 15000.00,
      "min_oi": 10000,
      "fees_roundtrip": 3.50,
      "hours": ["09:00", "18:00"],
      "after_market": ["18:00", "18:30"],
      "expiry_day": "First_Business_Day_Month",
      "slippage_base": {"high": 10.0, "avg": 2.0, "low": 1.0, "after": 15.0}
    },
    "note": "Dólar Cheio. Requer margem muito maior."
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

  "BIT$N": {
    "strategy": "MEAN_REVERSION",
    "active": True,
    "params": {
      "bb_period": 20,
      "bb_std": 2.50,
      "tp_mult": 3.50,
      "sl_atr_multiplier": 2.00,
      "adx_threshold": 25.0
    },
    "specs": {
      "tick_size": 10.0,
      "point_value": 0.10,
      "value_per_tick": 1.00,
      "margin": 300.00,
      "margin_stress": 5000.00,
      "min_oi": 1000,
      "fees_roundtrip": 1.50,
      "hours": ["09:00", "18:00"],
      "after_market": ["18:00", "18:30"],
      "expiry_day": "Last_Friday_Month",
      "slippage_base": {"high": 50.0, "avg": 20.0, "low": 10.0, "after": 100.0}
    },
    "note": "🚀 CRIPTO: Alta volatilidade, slippage ajustado para cima."
  },

  "ICF$N": {
    "strategy": "MEAN_REVERSION",
    "active": True,
    "params": {
      "bb_period": 22,
      "bb_std": 2.20,
      "tp_mult": 3.00,
      "sl_atr_multiplier": 1.50,
      "adx_threshold": 18.0
    },
    "specs": {
      "tick_size": 0.05,
      "point_value": 100.00,
      "value_per_tick": 5.00,
      "margin": 5000.00,
      "margin_stress": 10000.00,
      "min_oi": 500,
      "fees_roundtrip": 5.00,
      "hours": ["09:00", "16:00"],
      "after_market": None,
      "expiry_day": "Variable_Agricultural",
      "slippage_base": {"high": 0.20, "avg": 0.10, "low": 0.05, "after": 0.50}
    },
    "note": "☕ CAFÉ: Valor por tick em USD (convertido). Alavancagem alta."
  },

  "IND$N": {
    "strategy": "MEAN_REVERSION",
    "active": False,
    "params": {
      "bb_period": 20,
      "bb_std": 2.00
    },
    "specs": {
      "tick_size": 5.0,
      "point_value": 1.00,
      "value_per_tick": 5.00,
      "margin": 600.00,
      "margin_stress": 12500.00,
      "min_oi": 5000,
      "fees_roundtrip": 2.50,
      "hours": ["09:00", "17:55"],
      "after_market": ["18:00", "18:25"],
      "expiry_day": "Wednesday_closest_15th_Even_Month",
      "slippage_base": {"high": 25.0, "avg": 10.0, "low": 5.0, "after": 50.0}
    }
  }
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

