import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import utils
import config
from datetime import datetime

def run_stress_test(symbol, start_date, end_date):
    print(f"\n🔥 STRESS TEST: {symbol} | {start_date} até {end_date}")
    
    if not mt5.initialize():
        print("Erro ao inicializar MT5")
        return

    # Converte datas para datetime
    utc_from = datetime.strptime(start_date, "%Y-%m-%d")
    utc_to = datetime.strptime(end_date, "%Y-%m-%d")

    # Busca dados históricos
    rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M15, utc_from, utc_to)
    if rates is None or len(rates) == 0:
        print(f"❌ Sem dados para {symbol}")
        return

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    # Simulação simplificada:
    # 1. Calcula ATR médio da crise
    # 2. Verifica DD máximo
    # 3. Testa se o bot teria pausado (Circuit Breaker)
    
    close_prices = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    
    max_drawdown = 0
    peak = close_prices[0]
    for price in close_prices:
        if price > peak:
            peak = price
        dd = (peak - price) / peak
        if dd > max_drawdown:
            max_drawdown = dd
            
    avg_vol = df['close'].pct_change().std() * np.sqrt(252/96) # Volatilidade anualizada (96 barras de 15min/dia)
    
    print(f"📊 Max Drawdown: {max_drawdown:.2%}")
    print(f"📊 Volatilidade (15m): {avg_vol:.2%}")
    
    if max_drawdown > 0.05:
        print("⚠️ Crise Detectada: Drawdown > 5%")
    
    mt5.shutdown()

if __name__ == "__main__":
    # Exemplo: COVID Crash Março 2020
    run_stress_test("PETR4", "2020-03-01", "2020-04-01")
    # Exemplo: Joesley Day Maio 2017
    run_stress_test("IBOV", "2017-05-17", "2017-05-31")
