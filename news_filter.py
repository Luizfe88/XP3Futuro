# news_filter.py - Filtro de Notícias usando Calendário Econômico do MT5 + NewsAPI
"""
📰 NEWS FILTER - Integração Híbrida (MT5 + NewsAPI)
- Bloqueio dinâmico: High (2h), Medium (30min)
- Sentimento via NewsAPI + HuggingFace
"""

import MetaTrader5 as mt5
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import logging
import config
import utils
import numpy as np

logger = logging.getLogger("news_filter")

# Palavras-chave para sentimento (Brasil/B3) caso HF falhe
SENTIMENT_KEYWORDS = {
    "POSITIVO": ["lucro", "alta", "crescimento", "dividendo", "supera", "positivo", "recorde", "compra", "elevação", "acima", "otimismo"],
    "NEGATIVO": ["prejuízo", "queda", "baixa", "rebaixamento", "abaixo", "negativo", "crise", "risco", "venda", "corte", "pessimismo", "medo"]
}

# Cache interno
_news_cache: List[Dict] = []
_cache_timestamp: Optional[datetime] = None
CACHE_VALIDITY_MINUTES = 30
# Variável global para cache de sentimento
_sentiment_cache = {}

_sentiment_pipeline = None

def initialize_sentiment_engine():
    """Inicializa o modelo uma única vez no início do programa"""
    global _sentiment_pipeline
    if _sentiment_pipeline is not None:
        return

    try:
        from transformers import pipeline
        logger.info("🧠 Carregando motor de NLP (XLM-RoBERTa)...")
        # Pre-load model to avoid initialization lag during trading
        _sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
            device=-1 # Force CPU for thread safety
        )
        # Warmup
        _sentiment_pipeline("Teste de aquecimento")
        logger.info("✅ Motor NLP carregado com sucesso!")
    except Exception as e:
        logger.warning(f"⚠️ Falha ao carregar Transformers: {e}. Usando Fallback de Keywords.")
        _sentiment_pipeline = None

def get_huggingface_sentiment(text: str) -> Dict[str, float]:
    global _sentiment_pipeline, _sentiment_cache
    
    if not text:
        return {"negative": 0.0, "neutral": 1.0, "positive": 0.0}

    # 1. Check Cache
    hash_key = hash(text)
    if hash_key in _sentiment_cache:
        return _sentiment_cache[hash_key]

    # 2. Try Model Inference if available
    if _sentiment_pipeline is not None:
        try:
            # Truncate to 512 chars to avoid model errors
            results = _sentiment_pipeline(text[:512])[0]
            # Normalize scores if necessary, but pipeline usually returns probability
            # The model returns label ('positive', 'negative', 'neutral') and score
            # We need to map it correctly. 
            # Note: The specific model cardiffnlp/twitter-xlm-roberta-base-sentiment returns labels: 'positive', 'negative', 'neutral'
            
            # If the pipeline returns a list of dicts (top_k=None by default returns just the top class, we should probably use return_all_scores=True or top_k=None if we want full distribution, 
            # but standard pipeline returns just the top label if we don't specify)
            # Actually, let's stick to the user's existing logic structure but make it safe.
            # Assuming standard output: [{'label': 'positive', 'score': 0.9}]
            
            # To get full distribution, we should ideally call pipeline with return_all_scores=True (old) or top_k=None (new)
            # But let's look at the user's code: "scores = {res['label']: res['score'] for res in results}"
            # This implies the user expects a list of results.
            # Let's use robust extraction.
            
            # Re-running with top_k=None to get all scores
            full_results = _sentiment_pipeline(text[:512], top_k=None)
            # full_results is like [[{'label': 'positive', 'score': ...}, ...]]
            
            scores = {res['label']: res['score'] for res in full_results}
            
            result = {
                "negative": scores.get('negative', 0.0),
                "neutral": scores.get('neutral', 0.0),
                "positive": scores.get('positive', 0.0)
            }
            _sentiment_cache[hash_key] = result
            return result
        except Exception as e:
            logger.error(f"Erro na inferência ML: {e}")
            # Fallthrough to keywords
    
    # 3. Fallback: Keyword Logic
    text_lower = text.lower()
    neg_count = sum(1 for word in SENTIMENT_KEYWORDS["NEGATIVO"] if word in text_lower)
    pos_count = sum(1 for word in SENTIMENT_KEYWORDS["POSITIVO"] if word in text_lower)
    total = neg_count + pos_count
    
    if total == 0:
        result = {"negative": 0.0, "neutral": 1.0, "positive": 0.0}
    else:
        result = {
            "negative": neg_count / total, 
            "neutral": 0.0, 
            "positive": pos_count / total
        }
    
    _sentiment_cache[hash_key] = result
    return result
def fetch_from_newsapi(query: str, language: str = "pt") -> List[Dict]:
    """Busca notícias via NewsAPI."""
    api_key = getattr(config, "NEWS_API_KEY", "")
    if not api_key:
        return []
    
    try:
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "apiKey": api_key,
            "language": language,
            "sortBy": "publishedAt",
            "pageSize": 5,
            "from": (datetime.now() - timedelta(days=2)).isoformat()
        }
        resp = requests.get(url, params=params, timeout=5)
        data = resp.json()
        
        if data.get("status") == "ok":
            return [
                {"title": art["title"], "source": art["source"]["name"], "time": art["publishedAt"]}
                for art in data.get("articles", [])
            ]
        return []
    except Exception as e:
        logger.error(f"Erro NewsAPI: {e}")
        return []


def fetch_from_polygon_news(symbol: str) -> List[Dict]:
    """
    ✅ NOVO: Busca notícias via Polygon.io Reference API
    """
    try:
        ticker = symbol.replace(".SA", "")
        params = {
            "ticker": ticker,
            "limit": 5,
        }
        
        data = utils.get_polygon_data("v2/reference/news", params)
        if data:
            articles = data.get("results", [])
            logger.info(f"📰 Polygon News: {len(articles)} notícias para {symbol}")
            return [{"title": a.get("title", ""), "description": a.get("description", "")} for a in articles]
        return []
    except Exception as e:
        logger.error(f"Erro Polygon News: {e}")
        return []

def _fetch_mt5_calendar() -> List[Dict]:
    """Busca calendário econômico MT5 (Brasil)."""
    global _news_cache, _cache_timestamp
    now = datetime.now()
    
    if _cache_timestamp and (now - _cache_timestamp).seconds < CACHE_VALIDITY_MINUTES * 60:
        return _news_cache
    
    try:
        from_date = now - timedelta(hours=1)
        to_date = now + timedelta(hours=24)
        calendar_get = getattr(mt5, "calendar_get", None)
        events_raw = calendar_get(from_date, to_date) if callable(calendar_get) else None
        
        if not events_raw: 
            return []
        
        relevant_events = []
        for event in events_raw:
            if event.country != "BR": continue
            
            impact_map = {0: "Low", 1: "Medium", 2: "High"}
            impact = impact_map.get(event.importance, "Low")
            
            if impact == "Low": continue
            
            relevant_events.append({
                "time": datetime.fromtimestamp(event.time),
                "title": event.name.strip(),
                "impact": impact,
                "country": event.country
            })
        
        relevant_events.sort(key=lambda x: x["time"])
        _news_cache = relevant_events
        _cache_timestamp = now
        return relevant_events
    except Exception as e:
        logger.error(f"Erro calendário MT5: {e}")
        return _news_cache

def _check_fallback_windows(now: datetime | None = None) -> Tuple[bool, str]:
    if not getattr(config, "ENABLE_NEWS_FALLBACK_WINDOWS", True):
        return False, ""

    windows = getattr(config, "NEWS_FALLBACK_BLACKOUT_WINDOWS", None) or []
    now = now or datetime.now()
    now_time = now.time()

    for w in windows:
        start_str = (w or {}).get("start")
        end_str = (w or {}).get("end")
        label = (w or {}).get("label", "Janela de notícia")
        if not start_str or not end_str:
            continue

        try:
            start_time = datetime.strptime(start_str, "%H:%M").time()
            end_time = datetime.strptime(end_str, "%H:%M").time()
        except Exception:
            continue

        if start_time <= now_time <= end_time:
            return True, f"🚫 Blackout (fallback): {label} ({start_str}-{end_str})"

    return False, ""

_symbol_sentiment_state: Dict[str, Dict[str, object]] = {}

def _check_symbol_sentiment_blackout(symbol: str) -> Tuple[bool, str]:
    if not symbol or not getattr(config, "ENABLE_NEWS_SENTIMENT_BLOCK", False):
        return False, ""

    now = datetime.now()
    block_minutes = int(getattr(config, "NEWS_SENTIMENT_BLOCK_MINUTES", 60) or 60)
    threshold = float(getattr(config, "NEWS_SENTIMENT_NEG_THRESHOLD", -0.70) or -0.70)

    state = _symbol_sentiment_state.get(symbol, {})
    blocked_until = state.get("blocked_until")
    if isinstance(blocked_until, datetime) and blocked_until > now:
        return True, f"🚫 Blackout (sentimento): {symbol} até {blocked_until.strftime('%H:%M')}"

    last_check = state.get("last_check")
    if isinstance(last_check, datetime) and (now - last_check).total_seconds() < 15 * 60:
        return False, ""

    _symbol_sentiment_state[symbol] = {"last_check": now, "blocked_until": None}

    sentiment = get_news_sentiment(symbol)
    if sentiment <= threshold:
        until = now + timedelta(minutes=block_minutes)
        _symbol_sentiment_state[symbol] = {"last_check": now, "blocked_until": until}
        return True, f"🚫 Blackout (sentimento {sentiment:.2f}): {symbol} por {block_minutes}min"

    return False, ""

def get_upcoming_events(hours_ahead: int = 6) -> List[Dict]:
    events = _fetch_mt5_calendar()
    now = datetime.now()
    cutoff = now + timedelta(hours=hours_ahead)
    return [e for e in events if now <= e["time"] <= cutoff]

def check_news_blackout(symbol: str = "") -> Tuple[bool, str]:
    """
    Verifica blackout com janelas dinâmicas:
    - High: 2 horas antes
    - Medium: 30 min antes
    """
    if not getattr(config, "ENABLE_NEWS_FILTER", True):
        return False, ""
    
    try:
        upcoming = get_upcoming_events(hours_ahead=3)
        now = datetime.now()
        
        if not upcoming:
            fallback_block, fallback_reason = _check_fallback_windows(now)
            if fallback_block:
                return True, fallback_reason

            sentiment_block, sentiment_reason = _check_symbol_sentiment_blackout(symbol)
            if sentiment_block:
                return True, sentiment_reason

            return False, ""

        for event in upcoming:
            time_until = (event["time"] - now).total_seconds() / 60
            if time_until < 0: continue # Já passou
            
            # 🕒 Janelas Dinâmicas
            if event["impact"] == "High":
                block_window = 120 # 2 horas
            elif event["impact"] == "Medium":
                block_window = 30  # 30 minutos
            else:
                continue
            
            if time_until <= block_window:
                # Verifica sentimento para High Impact
                if event["impact"] == "High":
                    sentiment = get_huggingface_sentiment(event["title"])
                    if sentiment["negative"] > 0.6: # Risco alto
                        return True, f"🚫 Blackout High (Neg:{sentiment['negative']:.1%}): {event['title']} em {int(time_until)}min"
                    else:
                        logger.info(f"⚠️ Alerta High ({event['title']}), mas sentimento não é crítico.")
                        # Ainda bloqueia se for muito perto (<30min) mesmo sem sentimento negativo
                        if time_until < 30:
                            return True, f"🚫 Blackout High (Iminente): {event['title']} em {int(time_until)}min"
                else:
                    # Medium sempre bloqueia na janela curta
                    return True, f"🚫 Blackout Medium: {event['title']} em {int(time_until)}min"
                    
        return False, ""
        
    except Exception as e:
        logger.error(f"Erro news blackout: {e}")
        return False, ""

def get_next_high_impact_event() -> str:
    events = get_upcoming_events(hours_ahead=24)
    high = [e for e in events if e["impact"] == "High"]
    if high:
        mins = int((high[0]["time"] - datetime.now()).total_seconds() / 60)
        return f"Próximo High: {high[0]['title']} em {mins}min"
    return "Sem eventos High previstos"

def get_news_sentiment(symbol: str) -> float:
    """
    ✅ V5.2: Combina NewsAPI e Polygon.io News.
    Retorna score: -1 (muito negativo) a 1 (muito positivo).
    """
    try:
        # Busca notícias das duas fontes
        news_items = fetch_from_newsapi(symbol) + fetch_from_polygon_news(symbol)
        
        if not news_items:
            # Fallback para busca genérica se não achar pelo símbolo
            news_items = fetch_from_newsapi(f"{symbol} B3 mercado", language="pt")

        if news_items:
            sentiments = []
            for item in news_items[:8]:  # Analisa até 8 notícias recentes
                text = f"{item['title']} {item.get('description', '')}"
                hf_sentiment = get_huggingface_sentiment(text)
                # Score = Confiança Positiva - Confiança Negativa
                score = hf_sentiment["positive"] - hf_sentiment["negative"]
                sentiments.append(score)
            
            if sentiments:
                return float(np.mean(sentiments))

        return 0.0
    except Exception as e:
        logger.error(f"Erro sentimento {symbol}: {e}")
        return 0.0
