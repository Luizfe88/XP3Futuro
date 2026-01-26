# backtest.py
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import config
import utils
import time

# Configuração do Logger
logger = utils.setup_logger()

def run_backtest():
    logger.info("Iniciando Backtest...")
    
    # Período de 6 meses
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.DateOffset(months=6)
    
    results = []
    
    for asset in config.ASSETS:
        symbol_yf = f"{asset}.SA" # Adiciona sufixo para yfinance
        logger.info(f"Baixando dados para {symbol_yf}...")
        
        try:
            # Baixa dados
            df = yf.download(symbol_yf, start=start_date, end=end_date, interval=config.TIMEFRAME_STR, progress=False)
            
            if df.empty:
                logger.warning(f"Sem dados para {symbol_yf}")
                continue
                
            # Calcula Indicadores
            # EMA
            df['EMA_FAST'] = ta.ema(df['Close'], length=config.EMA_FAST)
            df['EMA_SLOW'] = ta.ema(df['Close'], length=config.EMA_SLOW)
            
            # RSI
            df['RSI'] = ta.rsi(df['Close'], length=config.RSI_PERIOD)
            
            # Volume MA
            df['VOL_MA'] = ta.sma(df['Volume'], length=config.VOLUME_MA_PERIOD)
            
            # Lógica de Trading Simplificada (Vectorized)
            # Condição de Compra:
            # 1. EMA9 > EMA21 (Cruzamento ou tendência) - Simplificado para estar acima
            # 2. RSI < 70
            # 3. Volume > Media Volume
            
            # Criar sinais
            df['Signal'] = 0
            buy_condition = (
                (df['EMA_FAST'] > df['EMA_SLOW']) & 
                (df['EMA_FAST'].shift(1) <= df['EMA_SLOW'].shift(1)) & # Cruzamento exato
                (df['RSI'] < config.RSI_OVERBOUGHT) & 
                (df['Volume'] > df['VOL_MA'])
            )
            
            df.loc[buy_condition, 'Signal'] = 1
            
            # Simulação de Trades (Iterativo para gerenciar TP/SL)
            trades = []
            position = None # {price, time, stop_loss, take_profit}
            
            for index, row in df.iterrows():
                # Se não tem posição, verifica compra
                if position is None:
                    if row['Signal'] == 1:
                        entry_price = row['Close'] # Simplificação: entra no fechamento do sinal
                        stop_loss = entry_price * (1 - config.STOP_LOSS_PCT)
                        take_profit = entry_price * (1 + config.TAKE_PROFIT_PCT)
                        
                        # Filtro RSI Stop Loss na entrada (opcional, mas descrito)
                        if row['RSI'] > config.RSI_STOP_LOSS:
                            continue
                            
                        position = {
                            'entry_price': entry_price,
                            'entry_time': index,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'highest_price': entry_price
                        }
                
                # Se tem posição, verifica saída
                else:
                    current_price = row['Close']
                    current_rsi = row['RSI']
                    
                    # Atualiza Trailing Stop (se preço subiu)
                    if current_price > position['highest_price']:
                        position['highest_price'] = current_price
                        new_stop = current_price * (1 - config.TRAILING_STOP_PCT)
                        if new_stop > position['stop_loss']:
                            position['stop_loss'] = new_stop
                    
                    # Condições de Saída
                    exit_reason = None
                    
                    # 1. Take Profit
                    if current_price >= position['take_profit']:
                        exit_reason = "TP"
                    
                    # 2. Stop Loss (Fixo ou Trailing)
                    elif current_price <= position['stop_loss']:
                        exit_reason = "SL"
                        
                    # 3. RSI Stop (RSI > 75)
                    elif current_rsi > config.RSI_STOP_LOSS:
                        exit_reason = "RSI_STOP"
                        
                    if exit_reason:
                        pnl_pct = (current_price - position['entry_price']) / position['entry_price']
                        trades.append({
                            'Asset': asset,
                            'Entry Time': position['entry_time'],
                            'Exit Time': index,
                            'Entry Price': position['entry_price'],
                            'Exit Price': current_price,
                            'PnL %': pnl_pct * 100,
                            'Reason': exit_reason
                        })
                        position = None
            
            # Consolida resultados do ativo
            if trades:
                df_trades = pd.DataFrame(trades)
                total_trades = len(df_trades)
                win_rate = len(df_trades[df_trades['PnL %'] > 0]) / total_trades * 100
                total_pnl = df_trades['PnL %'].sum()
                
                results.append({
                    'Asset': asset,
                    'Trades': total_trades,
                    'Win Rate %': round(win_rate, 2),
                    'Total PnL %': round(total_pnl, 2)
                })
                logger.info(f"{asset}: {total_trades} trades, Win Rate: {win_rate:.1f}%, PnL: {total_pnl:.1f}%")
            else:
                logger.info(f"{asset}: Nenhum trade gerado.")
                
        except Exception as e:
            logger.error(f"Erro no backtest de {asset}: {e}")
            
    # Relatório Final
    if results:
        df_results = pd.DataFrame(results)
        print("\n=== RELATÓRIO DE BACKTEST (6 MESES) ===")
        print(df_results.to_string(index=False))
        print("=======================================")
    else:
        print("Nenhum resultado gerado.")

if __name__ == "__main__":
    run_backtest()
