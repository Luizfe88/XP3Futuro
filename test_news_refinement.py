import MetaTrader5 as mt5
import logging
import news_filter
import config
import utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_news")

def test_news_refinement():
    if not mt5.initialize():
        print("Erro ao inicializar MT5")
        return

    symbol = "PETR4"
    
    print("\n--- Teste Sentiment Logic (HuggingFace/Fallback) ---")
    texts = [
        "Lucro da Petrobras sobe 50% apos corte de custos",
        "Prejuizo da Petrobras preocupa investidores",
        "Mercado lateral aguarda decisao do Copom"
    ]
    
    for text in texts:
        sent = news_filter.get_huggingface_sentiment(text)
        print(f"Texto: {text}")
        print(f"Sentiment: {sent}")

    print("\n--- Teste Blackout Refined ---")
    # Forçamos uma verificação. Se não houver eventos, o log dirá.
    blocked, reason = news_filter.check_news_blackout(symbol)
    print(f"Blackout Status: {blocked}")
    if blocked:
        print(f"Reason: {reason}")

    print("\n--- Teste R:R v5.2 (VIX > 30) ---")
    # Mocking get_vix_br to return 35
    original_vix = utils.get_vix_br
    utils.get_vix_br = lambda: 35.0
    
    rr = utils.get_dynamic_rr_min()
    print(f"Min R:R (VIX=35): {rr}")
    
    utils.get_vix_br = original_vix
    
    mt5.shutdown()

if __name__ == "__main__":
    test_news_refinement()
