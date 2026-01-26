import MetaTrader5 as mt5
import logging
from ml_signals import MLSignalPredictor
from ml_optimizer import EnsembleOptimizer
from fundamentals import fundamental_fetcher
from news_filter import get_news_sentiment
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_ml")

def test_ml_upgrade():
    if not mt5.initialize():
        print("Erro ao inicializar MT5")
        return

    symbol = "PETR4"
    
    print("\n--- Teste de Fundamentos ---")
    fund = fundamental_fetcher.get_fundamentals(symbol)
    print(f"PETR4 Fundamentals: {fund}")

    print("\n--- Teste de Sentimento ---")
    sentiment = get_news_sentiment(symbol)
    print(f"PETR4 Sentiment Score: {sentiment}")

    print("\n--- Teste de MLPredictor (com Imbalance) ---")
    predictor = MLSignalPredictor()
    indicators = {
        'rsi': 30.0,
        'adx': 30.0,
        'atr_pct': 1.0,
        'volume_ratio': 1.5,
        'momentum': 0.01,
        'ema_diff': 0.05,
        'macd': 0.02,
        'price_vs_vwap': 0.01,
        'vwap': 35.0,
        'close': 35.5
    }
    
    # Testa predição
    # Nota: Como o book é real-time, pode variar se o mercado estiver aberto ou fechado.
    prediction = predictor.predict(symbol, indicators)
    print(f"Predição PETR4: {prediction}")

    print("\n--- Teste de MLOptimizer (Q-States e CV) ---")
    optimizer = EnsembleOptimizer()
    print(f"Q-Table Stats: Shape={optimizer.q_table.shape}")
    
    # Testa discretização
    state = optimizer.discretize_state(indicators)
    print(f"Discretized State: {state} (Max: {optimizer.states})")
    
    # Testa extração de features
    features = optimizer.extract_features(indicators, symbol)
    print(f"Features extraídas (keys): {list(features.keys())}")
    
    mt5.shutdown()

if __name__ == "__main__":
    test_ml_upgrade()
