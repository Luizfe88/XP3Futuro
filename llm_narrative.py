"""
llm_narrative.py - Geração de Narrativas de Mercado via IA
🤖 Integração com GPT-4o / Claude para explicar decisões de trading.
"""

import os
import requests
import json
import logging
import config
from typing import Dict, Any

logger = logging.getLogger("llm")

def generate_market_narrative(symbol: str, indicators: Dict[str, Any], sentiment: float) -> str:
    """
    ✅ Gera uma narrativa curta sobre as condições atuais do mercado.
    Baseado em indicadores técnicos e sentimento de notícias.
    """
    api_key = getattr(config, "LLM_API_KEY", "")
    provider = getattr(config, "LLM_PROVIDER", "openai") # 'openai' ou 'anthropic'
    
    if not api_key or api_key == "MOCK_KEY":
        return "🤖 LLM: Narrativa indisponível (chave não configurada)."

    try:
        # Prompt contextualizado
        prompt = f"""
        Analise o ativo {symbol} para trading intradiário (M15):
        - Técnica: RSI={indicators.get('rsi', 50):.1f}, ADX={indicators.get('adx', 20):.1f}, Vol={indicators.get('volume_ratio', 1.0):.1f}x.
        - Sentimento: {sentiment:+.2f} (escala -1 a 1).
        
        Dê um resumo técnico de 2 frases focado na força da tendência e fluxo. Seja direto e profissional.
        """

        if provider == "openai":
            url = "https://api.openai.com/v1/chat/completions"
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            payload = {
                "model": "gpt-4o-mini", # Usando mini para velocidade/custo
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 100
            }
        else:
            # Placeholder para Anthropic/Claude
            return "🤖 LLM (Claude): Provedor em implementação."

        response = requests.post(url, headers=headers, json=payload, timeout=10)
        if response.status_code == 200:
            narrative = response.json()['choices'][0]['message']['content'].strip()
            return f"🤖 LLM: {narrative}"
        
        return "🤖 LLM: Erro na conexão com a API."

    except Exception as e:
        logger.error(f"Erro ao gerar narrativa LLM ({symbol}): {e}")
        return "🤖 LLM: Instabilidade momentânea na análise."

if __name__ == "__main__":
    # Teste
    dummy_ind = {"rsi": 32, "adx": 35, "volume_ratio": 2.1}
    print(generate_market_narrative("PETR4", dummy_ind, 0.45))
