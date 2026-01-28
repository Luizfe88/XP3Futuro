import yfinance as yf
import logging
import json
import os
from datetime import datetime, timedelta
from typing import Tuple, Dict

logger = logging.getLogger("fundamentals")

# Limites configuráveis para filtro de tradeabilidade
FUNDAMENTAL_LIMITS = {
    "pe_min": 0,        # P/L mínimo (evita empresas com prejuízo)
    "pe_max": 30,       # P/L máximo (evita empresas muito caras)
    "roe_min": 0.05,    # ROE mínimo de 5%
    "market_cap_min": 1e9  # Market cap mínimo de R$ 1 bilhão
}

class FundamentalFetcher:
    """
    Busca dados fundamentistas (P/L, ROE) via yfinance.
    Utiliza cache local para evitar excesso de requisições.
    """
    def __init__(self, cache_file="fundamentals_cache.json"):
        self.cache_file = cache_file
        self.cache = self._load_cache()
        self.cache_validity_days = 7  # Dados fundamentistas mudam pouco

    def _load_cache(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Erro ao carregar cache fundamentista: {e}")
        return {}

    def _save_cache(self):
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f)
        except Exception as e:
            logger.error(f"Erro ao salvar cache fundamentista: {e}")

    def get_fundamentals(self, symbol: str) -> dict:
        """
        Retorna P/L e ROE para um símbolo da B3.
        """
        # Formata para padrão Yahoo (ex: PETR4.SA)
        if not symbol.endswith(".SA"):
            yf_symbol = f"{symbol}.SA"
        else:
            yf_symbol = symbol

        now = datetime.now()
        
        # Verifica cache
        if yf_symbol in self.cache:
            entry = self.cache[yf_symbol]
            updated_at = datetime.strptime(entry['updated_at'], "%Y-%m-%d %H:%M:%S")
            if (now - updated_at).days < self.cache_validity_days:
                return entry['data']

        try:
            logger.info(f"🔍 Buscando dados fundamentistas para {yf_symbol}...")
            ticker = yf.Ticker(yf_symbol)
            info = ticker.info
            
            data = {
                "pe_ratio": info.get("trailingPE", 0.0) or 0.0,
                "forward_pe": info.get("forwardPE", 0.0) or 0.0,
                "roe": info.get("returnOnEquity", 0.0) or 0.0,
                "roa": info.get("returnOnAssets", 0.0) or 0.0,
                "market_cap": info.get("marketCap", 0.0) or 0.0,
                "dividend_yield": info.get("dividendYield", 0.0) or 0.0,
                "debt_to_equity": info.get("debtToEquity", 0.0) or 0.0,
                "current_ratio": info.get("currentRatio", 0.0) or 0.0
            }
            
            # Atualiza cache
            self.cache[yf_symbol] = {
                "updated_at": now.strftime("%Y-%m-%d %H:%M:%S"),
                "data": data
            }
            self._save_cache()
            
            return data
        except Exception as e:
            logger.error(f"Erro ao buscar fundamentos para {yf_symbol}: {e}")
            return {"pe_ratio": 0.0, "roe": 0.0, "market_cap": 0.0}

    def check_tradeability(self, symbol: str) -> Tuple[bool, str]:
        """
        Verifica se o ativo é tradeable baseado em P/L e ROE.
        
        Returns:
            (is_tradeable, reason)
        """
        data = self.get_fundamentals(symbol)
        
        pe = data.get("pe_ratio", 0)
        roe = data.get("roe", 0)
        market_cap = data.get("market_cap", 0)
        
        # P/L negativo ou zero indica prejuízo
        if pe <= FUNDAMENTAL_LIMITS["pe_min"]:
            return False, f"P/L negativo ou zero ({pe:.1f})"
        
        # P/L muito alto indica sobrevalorização
        if pe > FUNDAMENTAL_LIMITS["pe_max"]:
            return False, f"P/L muito alto ({pe:.1f} > {FUNDAMENTAL_LIMITS['pe_max']})"
        
        # ROE mínimo
        if roe < FUNDAMENTAL_LIMITS["roe_min"]:
            return False, f"ROE baixo ({roe:.1%} < {FUNDAMENTAL_LIMITS['roe_min']:.1%})"
        
        # Market cap mínimo (liquidez)
        if market_cap < FUNDAMENTAL_LIMITS["market_cap_min"]:
            return False, f"Market Cap baixo (R$ {market_cap/1e6:.0f}M)"
        
        return True, f"OK (P/L: {pe:.1f}, ROE: {roe:.1%})"

    def is_fundamentally_sound(self, symbol: str) -> bool:
        """
        Retorna True se o ativo passa nos filtros fundamentalistas.
        """
        tradeable, _ = self.check_tradeability(symbol)
        return tradeable

    def get_fundamental_score(self, symbol: str) -> float:
        """
        Calcula um score fundamentalista de 0 a 100.
        """
        data = self.get_fundamentals(symbol)
        
        pe = data.get("pe_ratio", 0)
        roe = data.get("roe", 0)
        
        score = 50  # Base
        
        # P/L ideal entre 5-15
        if 5 <= pe <= 15:
            score += 25
        elif 15 < pe <= 25:
            score += 10
        elif pe > 25 or pe <= 0:
            score -= 15
        
        # ROE bonus
        if roe >= 0.20:
            score += 25
        elif roe >= 0.10:
            score += 15
        elif roe >= 0.05:
            score += 5
        else:
            score -= 10
        
        return max(0, min(100, score))


# Instância global
fundamental_fetcher = FundamentalFetcher()

if __name__ == "__main__":
    # Teste rápido
    logging.basicConfig(level=logging.INFO)
    
    for sym in ["PETR4", "VALE3", "ITUB4", "MGLU3"]:
        data = fundamental_fetcher.get_fundamentals(sym)
        tradeable, reason = fundamental_fetcher.check_tradeability(sym)
        score = fundamental_fetcher.get_fundamental_score(sym)
        
        print(f"\n{sym}:")
        print(f"  P/L: {data['pe_ratio']:.1f} | ROE: {data['roe']:.1%}")
        print(f"  Tradeable: {tradeable} ({reason})")
        print(f"  Score: {score:.0f}/100")

