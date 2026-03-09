import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import logging
import time
from datetime import datetime
import os

logger = logging.getLogger("ExecutionEngine")

class ExecutionEngine:
    """
    Motor de execução para MetaTrader 5.
    Gerencia ordens, streaming de dados e log de slippage.
    """
    def __init__(self, symbol="WIN$N", magic_number=123456):
        self.symbol = symbol
        self.magic = magic_number
        self.slippage_log_path = "logs/slippage_analysis.csv"
        os.makedirs("logs", exist_ok=True)
        
        if not os.path.exists(self.slippage_log_path):
            df_init = pd.DataFrame(columns=["timestamp", "symbol", "signal_price", "exec_price", "slippage_points", "type"])
            df_init.to_csv(self.slippage_log_path, index=False)

    def connect(self, path=r"C:\MetaTrader 5 Terminal\terminal64.exe"):
        if not mt5.initialize(path=path):
            logger.error(f"[FAIL] Falha ao inicializar MT5 no caminho: {path}")
            return False
        logger.info(f"[OK] MT5 Conectado. Ativo: {self.symbol}")
        return True

    def get_latest_m1_data(self, count=20):
        """Busca os últimos candles M1 para atualizar o Kalman."""
        rates = mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_M1, 0, count)
        if rates is None or len(rates) == 0:
            return None
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df

    def execute_market_order(self, type, volume, signal_price, sl_points=None):
        """
        Executa uma ordem a mercado com Stop Loss dinâmico.
        sl_points: Distância do Stop Loss em pontos.
        """
        tick = mt5.symbol_info_tick(self.symbol)
        price_request = tick.ask if type == mt5.ORDER_TYPE_BUY else tick.bid
        
        # Cálculo de Stop Loss (Proteção Inicial)
        sl_price = 0.0
        if sl_points:
            sl_price = price_request - sl_points if type == mt5.ORDER_TYPE_BUY else price_request + sl_points

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": float(volume),
            "type": type,
            "price": price_request,
            "sl": float(sl_price) if sl_price > 0 else 0.0,
            "magic": self.magic,
            "comment": "Quant - Shadow Trading",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }

        result = mt5.order_send(request)
        
        if result is None:
            logger.error(f"[FAIL] Erro crítico no order_send: {mt5.last_error()}")
            return None

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.warning(f"[WARN] Ordem rejeitada! Retcode: {result.retcode}")
            return result

        # Log de Slippage
        exec_price = result.price
        slippage = abs(exec_price - signal_price)
        self._log_slippage(signal_price, exec_price, type)
        
        logger.info(f"[ORDER] {self.symbol} | Exec: {exec_price} | SL: {sl_price:.2f} | Slippage: {slippage:.2f}")
        return result

    def _log_slippage(self, signal_price, exec_price, order_type):
        """Grava dados para análise de latência e custo real."""
        new_row = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": self.symbol,
            "signal_price": signal_price,
            "exec_price": exec_price,
            "slippage_points": abs(exec_price - signal_price),
            "type": "BUY" if order_type == mt5.ORDER_TYPE_BUY else "SELL"
        }
        df = pd.DataFrame([new_row])
        df.to_csv(self.slippage_log_path, mode='a', header=False, index=False)

    def close_all_positions(self):
        """Fecha todas as posições abertas pelo robô (Magic Number)."""
        positions = mt5.positions_get(magic=self.magic)
        if positions:
            for pos in positions:
                tick = mt5.symbol_info_tick(pos.symbol)
                type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
                price = tick.bid if type == mt5.ORDER_TYPE_SELL else tick.ask
                
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": pos.symbol,
                    "pos": pos.ticket,
                    "volume": pos.volume,
                    "type": type,
                    "price": price,
                    "magic": self.magic,
                    "comment": "Quant - Close All",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_FOK,
                }
                mt5.order_send(request)
            logger.info(f"🧹 {len(positions)} posições fechadas.")

    def shutdown(self):
        """Finaliza a conexão com o MT5."""
        mt5.shutdown()
        logger.info("[MT5] Conexão MT5 encerrada.")
