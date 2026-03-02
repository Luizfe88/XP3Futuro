# validation package
"""Wrapper package exposing validation functionality.

The original implementation was located in a top-level module
``validation.py``.  To avoid a naming collision with this package we
moved the implementation into ``validation_core.py`` and then re-exported the
public API here.  Existing code can continue to import ``validation`` or use
``from validation import ...`` as before.
"""

from __future__ import annotations

# Prefer explicit exports so that consumers of ``validation`` always see
# the expected API symbols even if the wildcard import mechanism misbehaves.
# We still use a try/except in case the package isn't installed, allowing
# tests or scripts executed directly from the repository root to import
# ``validation`` without having xp3future on sys.path.
from typing import TYPE_CHECKING

try:
    from xp3future.validation_core import (
        validate_and_create_order,
        OrderParams,
        OrderSide,
        register_stop_loss,
        check_revenge_cooldown,
        validate_spread_protection,
        check_daily_loss_money_limit,
        check_capital_usage_limit,
        validate_subsetor_exposure,
        monte_carlo_ruin_check,
        calculate_kelly_position_size,
        # add any other public helpers you expect external code to use
    )
except ImportError:
    # fallback when running from repo without package installation
    from validation_core import (
        validate_and_create_order,
        OrderParams,
        OrderSide,
        register_stop_loss,
        check_revenge_cooldown,
        validate_spread_protection,
        check_daily_loss_money_limit,
        check_capital_usage_limit,
        validate_subsetor_exposure,
        monte_carlo_ruin_check,
        calculate_kelly_position_size,
    )

# Export symbols for ``from validation import *`` callers
__all__ = [
    "validate_and_create_order",
    "OrderParams",
    "OrderSide",
    "register_stop_loss",
    "check_revenge_cooldown",
    "validate_spread_protection",
    "check_daily_loss_money_limit",
    "check_capital_usage_limit",
    "validate_subsetor_exposure",
    "monte_carlo_ruin_check",
    "calculate_kelly_position_size",
]
