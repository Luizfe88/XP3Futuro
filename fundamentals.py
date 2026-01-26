import logging
import json
import os
from datetime import datetime, timedelta
from typing import Tuple, Dict
import MetaTrader5 as mt5
import pandas as pd
import config

logger = logging.getLogger("fundamentals")

# Limites configuráveis para filtro de tradeabilidade (MT5-only)
FUNDAMENTAL_LIMITS = {
    "min_avg_tick_volume": 1.0,
    "min_atr_pct": 0.005
}

class FundamentalFetcher:
    """
    Coleta métricas essenciais via MT5 (sem dependência do Yahoo).
    Usa cache leve para evitar leituras repetidas quando possível.
    """
    def __init__(self, cache_file="fundamentals_cache.json"):
        self.cache_file = cache_file
        self.cache = self._load_cache()
        self.cache_validity_minutes = 60

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
        now = datetime.now()
        cache_key = symbol
        if cache_key in self.cache:
            entry = self.cache[cache_key]
            updated_at = datetime.strptime(entry['updated_at'], "%Y-%m-%d %H:%M:%S")
            if (now - updated_at) <= timedelta(minutes=self.cache_validity_minutes):
                return entry['data']

        mt5_symbol = symbol.replace(".SA", "") if symbol.endswith(".SA") else symbol
        try:
            path = getattr(config, "MT5_TERMINAL_PATH", None)
            if path:
                mt5.initialize(path=path)
            else:
                mt5.initialize()
        except Exception:
            pass

        try:
            try:
                mt5.symbol_select(mt5_symbol, True)
            except Exception:
                pass
            rates = mt5.copy_rates_from_pos(mt5_symbol, mt5.TIMEFRAME_D1, 0, 300)
            if not rates:
                data = {"mt5_bars": 0, "mt5_avg_tick_volume": 0.0, "mt5_atr_pct": 0.0}
            else:
                df = pd.DataFrame(rates)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                df.set_index('time', inplace=True)
                avg_vol = float(df['tick_volume'].mean()) if 'tick_volume' in df.columns else float(df['volume'].mean()) if 'volume' in df.columns else 0.0
                hl_range = (df['high'] - df['low']).mean() if {'high','low'}.issubset(df.columns) else 0.0
                close_mean = df['close'].mean() if 'close' in df.columns else 0.0
                atr_pct = float(hl_range / close_mean) if close_mean else 0.0
                data = {
                    "mt5_bars": int(len(df)),
                    "mt5_avg_tick_volume": avg_vol,
                    "mt5_atr_pct": atr_pct
                }
            try:
                mt5.shutdown()
            except Exception:
                pass

            self.cache[cache_key] = {
                "updated_at": now.strftime("%Y-%m-%d %H:%M:%S"),
                "data": data
            }
            self._save_cache()
            return data
        except Exception as e:
            logger.error(f"Erro ao coletar métricas MT5 para {mt5_symbol}: {e}")
            try:
                mt5.shutdown()
            except Exception:
                pass
            return {"mt5_bars": 0, "mt5_avg_tick_volume": 0.0, "mt5_atr_pct": 0.0}

    def check_tradeability(self, symbol: str) -> Tuple[bool, str]:
        data = self.get_fundamentals(symbol)
        avg_vol = data.get("mt5_avg_tick_volume", 0.0)
        atr_pct = data.get("mt5_atr_pct", 0.0)
        bars = data.get("mt5_bars", 0)
        if bars <= 0:
            return False, "Sem histórico no MT5"
        if avg_vol < FUNDAMENTAL_LIMITS["min_avg_tick_volume"]:
            return False, "Sem liquidez MT5"
        if atr_pct < FUNDAMENTAL_LIMITS["min_atr_pct"]:
            return False, "Volatilidade insuficiente MT5"
        return True, f"OK MT5 (vol:{avg_vol:.0f} atr%:{atr_pct:.3f})"

    def is_fundamentally_sound(self, symbol: str) -> bool:
        """
        Retorna True se o ativo passa nos filtros fundamentalistas.
        """
        tradeable, _ = self.check_tradeability(symbol)
        return tradeable

    def get_fundamental_score(self, symbol: str) -> float:
        data = self.get_fundamentals(symbol)
        avg_vol = data.get("mt5_avg_tick_volume", 0.0)
        atr_pct = data.get("mt5_atr_pct", 0.0)
        bars = data.get("mt5_bars", 0)
        s = 40.0
        if bars < 100:
            s -= 10
        if avg_vol > 1000:
            s += 25
        elif avg_vol > 200:
            s += 10
        else:
            s -= 10
        if atr_pct > 0.02:
            s += 25
        elif atr_pct > 0.01:
            s += 10
        else:
            s -= 10
        return max(0, min(100, s))


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
        print(f"  MT5 barras: {data.get('mt5_bars', 0)} | vol médio: {data.get('mt5_avg_tick_volume', 0.0):.0f} | ATR%: {data.get('mt5_atr_pct', 0.0):.3f}")
        print(f"  Tradeable: {tradeable} ({reason})")
        print(f"  Score: {score:.0f}/100")

