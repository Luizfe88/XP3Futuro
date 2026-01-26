import MetaTrader5 as mt5
import logging
from validation import OrderParams, OrderSide, calculate_kelly_position_size, monte_carlo_ruin_check
from hedging import apply_hedge, check_unwind_trigger
import config
import utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_risk")

def test_risk_upgrade():
    if not mt5.initialize():
        print("Erro ao inicializar MT5")
        return

    symbol = "PETR4"
    
    print("\n--- Teste Monte Carlo Ruin Check ---")
    prob_ruin = monte_carlo_ruin_check(win_rate=0.55, rr=1.5, fraction=0.1)
    print(f"Ruin Prob (WR=55%, RR=1.5, Kelly=10%): {prob_ruin:.1%}")

    print("\n--- Teste Kelly Volume with Monte Carlo ---")
    volume = calculate_kelly_position_size(symbol, 35.0, 34.0, 37.0, "BUY")
    print(f"Volume Kelly para PETR4: {volume}")

    print("\n--- Teste OrderParams (R:R, ATR%, IBOV Corr) ---")
    try:
        order = OrderParams(
            symbol=symbol,
            side=OrderSide.BUY,
            volume=100,
            entry_price=35.0,
            sl=34.5, # RR = 2.0 / 0.5 = 4.0
            tp=37.0
        )
        print(f"Ordem validada: RR={order.risk_reward_ratio}")
    except Exception as e:
        print(f"Erro validação ordem: {e}")

    print("\n--- Teste Unwind Hedge Trigger ---")
    # Simula DD baixo e VIX baixo para trigger de unwind
    check_unwind_trigger(current_dd=0.01, vix_br=20.0)
    print("Check unwind executado (verificar logs)")

    print("\n--- Teste Busca Opções PUT ---")
    from hedging import hedge_with_options
    hedge_with_options(symbol)
    print("Busca de opções executada (verificar logs)")

    mt5.shutdown()

if __name__ == "__main__":
    test_risk_upgrade()
