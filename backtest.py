# backtest.py - Módulo de Backtesting Integrado para XP3 Bot
"""
📊 Sistema de backtesting para simulação de trades históricos
✅ Calcula Win Rate, Profit Factor, e Drawdown
✅ Usa dados reais do MT5 (history_deals_get)
✅ Integrado com o ciclo diário do bot
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
import logging
import config

logger = logging.getLogger("backtest")


def run_backtest(days: int = 30) -> Optional[float]:
    """
    Executa backtesting nos últimos N dias usando dados reais do MT5.
    
    Args:
        days: Número de dias para análise
        
    Returns:
        Win Rate (%) ou None se sem dados
    """
    logger.info(f"📊 Iniciando Backtest ({days} dias)...")
    
    # Define período
    utc_to = datetime.now()
    utc_from = utc_to - timedelta(days=days)
    
    # Busca histórico de deals
    deals = mt5.history_deals_get(utc_from, utc_to)
    
    if deals is None or len(deals) == 0:
        logger.warning("⚠️ Nenhum deal encontrado no período")
        return None
    
    df = pd.DataFrame(list(deals), columns=deals[0]._asdict().keys())
    
    # Filtra apenas fechamentos (DEAL_ENTRY_OUT)
    df_out = df[df['entry'] == mt5.DEAL_ENTRY_OUT].copy()
    
    if len(df_out) == 0:
        logger.warning("⚠️ Nenhum deal de saída encontrado")
        return None
    
    # Calcula métricas
    total_trades = len(df_out)
    wins = len(df_out[df_out['profit'] > 0])
    losses = len(df_out[df_out['profit'] <= 0])
    
    win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
    
    total_profit = df_out[df_out['profit'] > 0]['profit'].sum()
    total_loss = abs(df_out[df_out['profit'] <= 0]['profit'].sum())
    
    profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
    
    # Calcula Drawdown
    df_out['cumulative_pnl'] = df_out['profit'].cumsum()
    df_out['peak'] = df_out['cumulative_pnl'].cummax()
    df_out['drawdown'] = df_out['peak'] - df_out['cumulative_pnl']
    max_drawdown = df_out['drawdown'].max()
    
    # Log resultados
    logger.info(f"📊 Resultados Backtest ({days} dias):")
    logger.info(f"   Total Trades: {total_trades}")
    logger.info(f"   Wins: {wins} | Losses: {losses}")
    logger.info(f"   Win Rate: {win_rate:.1f}%")
    logger.info(f"   Profit Factor: {profit_factor:.2f}")
    logger.info(f"   Max Drawdown: R$ {max_drawdown:,.2f}")
    
    return win_rate


        rsi_low: RSI de compra
        rsi_high: RSI de venda
        df: DataFrame opcional (se None, baixa dados novos)
        
    Returns:
        Dict com resultados da simulação
    """
    logger.info(f"🔬 Simulando {symbol} ({days if df is None else len(df)} barras)...")
    
    if df is None:
        # Busca dados históricos
        utc_to = datetime.now()
        utc_from = utc_to - timedelta(days=days)
        rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M15, utc_from, utc_to)
        
        if rates is None or len(rates) < ema_long + 20:
            return {"error": "Dados insuficientes"}
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
    else:
        df = df.copy()

    if len(df) < ema_long + 20:
        return {"error": "DF muito curto para simulação"}
    
    # Calcula indicadores
    df['ema_short'] = df['close'].ewm(span=ema_short, adjust=False).mean()
    df['ema_long'] = df['close'].ewm(span=ema_long, adjust=False).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Simula sinais
    df['signal'] = 0
    df.loc[(df['ema_short'] > df['ema_long']) & (df['rsi'] < rsi_high), 'signal'] = 1  # BUY
    df.loc[(df['ema_short'] < df['ema_long']) & (df['rsi'] > rsi_low), 'signal'] = -1  # SELL
    
    # Remove primeiras linhas (indicadores ainda não calculados)
    df = df.iloc[ema_long + 20:].copy()
    
    # Simula trades
    trades = []
    position = None
    
    for i, row in df.iterrows():
        if position is None:
            # Abre posição
            slippage = 0.0005  # 0.05% de slippage médio
            
            if row['signal'] == 1:
                # Buy com slippage (paga mais caro)
                entry_price = row['close'] * (1 + slippage)
                position = {'side': 'BUY', 'entry': entry_price, 'time': row['time']}
            elif row['signal'] == -1:
                # Sell com slippage (vende mais barato)
                entry_price = row['close'] * (1 - slippage)
                position = {'side': 'SELL', 'entry': entry_price, 'time': row['time']}
        else:
            # Fecha posição
            slippage = 0.0005  # 0.05% de slippage na saída
            
            if position['side'] == 'BUY' and row['signal'] == -1:
                exit_price = row['close'] * (1 - slippage)
                pnl = exit_price - position['entry']
                trades.append({'pnl': pnl, 'side': 'BUY', 'duration': row['time'] - position['time']})
                position = None
            elif position['side'] == 'SELL' and row['signal'] == 1:
                exit_price = row['close'] * (1 + slippage)
                pnl = position['entry'] - exit_price
                trades.append({'pnl': pnl, 'side': 'SELL', 'duration': row['time'] - position['time']})
                position = None
    
    if not trades:
        return {"error": "Nenhum trade simulado"}
    
    # Calcula métricas
    df_trades = pd.DataFrame(trades)
    wins = len(df_trades[df_trades['pnl'] > 0])
    total = len(df_trades)
    win_rate = (wins / total) * 100
    
    total_pnl = df_trades['pnl'].sum()
    avg_win = df_trades[df_trades['pnl'] > 0]['pnl'].mean() if wins > 0 else 0
    avg_loss = df_trades[df_trades['pnl'] <= 0]['pnl'].mean() if total - wins > 0 else 0
    
    results = {
        "symbol": symbol,
        "period_days": days,
        "total_trades": total,
        "wins": wins,
        "losses": total - wins,
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "rr_ratio": abs(avg_win / avg_loss) if avg_loss != 0 else 0
    }
    
    logger.info(f"📊 Simulação {symbol}: WR={win_rate:.1f}%, Trades={total}, PnL={total_pnl:.2f}")
    
    return results


    return True, f"Parâmetros válidos (WR: {results['win_rate']:.1f}%, R:R: {results['rr_ratio']:.2f})"


def run_walk_forward(symbol: str, days: int = 60, segments: int = 4) -> List[Dict]:
    """
    ✅ Implementa Walk-Forward Optimization (80/20 train/test split).
    Divide os dados em segmentos e valida a robustez da estratégia.
    """
    logger.info(f"🚀 Iniciando Walk-Forward ({segments} segmentos) para {symbol}...")
    
    # Busca dados totais
    from utils import safe_copy_rates
    df_total = safe_copy_rates(symbol, mt5.TIMEFRAME_M15, days * 100)
    
    if df_total is None or len(df_total) < 500:
        logger.error("❌ Dados insuficientes para WFO")
        return []

    segment_size = len(df_total) // segments
    wfo_results = []

    for i in range(segments):
        win_start = i * segment_size
        win_end = (i + 1) * segment_size
        df_window = df_total.iloc[win_start:win_end]
        
        # Split 80/20
        split_point = int(0.8 * len(df_window))
        train_df = df_window.iloc[:split_point]
        test_df = df_window.iloc[split_point:]
        
        # Otimização básica (Busca melhor EMA no treino)
        best_ema = 9
        best_wr = 0
        for ema in [9, 12, 15, 21]:
            res = simulate_strategy(symbol, ema_short=ema, df=train_df)
            if "error" not in res and res['win_rate'] > best_wr:
                best_wr = res['win_rate']
                best_ema = ema
        
        # Validação no Teste (Out-of-Sample)
        oos_perf = simulate_strategy(symbol, ema_short=best_ema, df=test_df)
        
        if "error" not in oos_perf:
            wfo_results.append({
                "segment": i + 1,
                "best_ema_train": best_ema,
                "train_wr": best_wr,
                "test_wr": oos_perf['win_rate'],
                "test_trades": oos_perf['total_trades'],
                "is_robust": oos_perf['win_rate'] > 50
            })
            logger.info(f"✅ WFO Segmento {i+1}: Train WR={best_wr:.1f}% | Test WR={oos_perf['win_rate']:.1f}%")

    return wfo_results


if __name__ == "__main__":
    if not mt5.initialize():
        print("❌ Erro ao inicializar MT5")
    else:
        # Teste rápido
        wr = run_backtest(30)
        print(f"\n✅ Win Rate (30 dias): {wr:.1f}%" if wr else "Sem dados")
        
        # Simula estratégia
        results = simulate_strategy("PETR4", days=30)
        if "error" not in results:
            print(f"\n📊 Simulação PETR4:")
            for k, v in results.items():
                print(f"   {k}: {v}")
        
        mt5.shutdown()
