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
        
        # Sincroniza informações do símbolo
        if not mt5.symbol_select(self.symbol, True):
            logger.error(f"[FAIL] Falha ao selecionar símbolo: {self.symbol}")
            return False
            
        logger.info(f"[OK] MT5 Conectado. Ativo: {self.symbol}")
        return True

    def normalize_price(self, price):
        """Ajusta o preço para o tick size e número de dígitos do ativo."""
        info = mt5.symbol_info(self.symbol)
        if info is None: return price
        
        tick_size = info.trade_tick_size
        if tick_size > 0:
            rounded = round(price / tick_size) * tick_size
            return round(rounded, info.digits)
        return round(price, info.digits)

    def get_latest_m1_data(self, count=20):
        """Busca os últimos candles M1 para atualizar o Kalman."""
        rates = mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_M1, 0, count)
        if rates is None or len(rates) == 0:
            return None
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df

    def get_latest_data(self, timeframe, count=20):
        """Busca os últimos candles de um timeframe específico."""
        rates = mt5.copy_rates_from_pos(self.symbol, timeframe, 0, count)
        if rates is None or len(rates) == 0:
            return None
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df

    def execute_market_order(self, type, volume, signal_price, sl_points=None, tp_points=None):
        """
        Executa uma ordem a mercado com Stop Loss e Take Profit dinâmicos.
        sl_points: Distância do Stop Loss em pontos.
        tp_points: Distância do Take Profit em pontos.
        """
        tick = mt5.symbol_info_tick(self.symbol)
        price_request = tick.ask if type == mt5.ORDER_TYPE_BUY else tick.bid
        
        # Cálculo de Stop Loss (Proteção Inicial)
        sl_price = 0.0
        if sl_points:
            sl_price = price_request - sl_points if type == mt5.ORDER_TYPE_BUY else price_request + sl_points
            sl_price = self.normalize_price(sl_price)

        # Cálculo de Take Profit
        tp_price = 0.0
        if tp_points:
            tp_price = price_request + tp_points if type == mt5.ORDER_TYPE_BUY else price_request - tp_points
            tp_price = self.normalize_price(tp_price)

        # Normaliza preço de entrada se necessário
        price_request = self.normalize_price(price_request)

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": float(volume),
            "type": type,
            "price": float(price_request),
            "sl": float(sl_price) if sl_price > 0 else 0.0,
            "tp": float(tp_price) if tp_price > 0 else 0.0,
            "magic": self.magic,
            "comment": "Quant - Shadow Trading",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC, # Alterado para IOC (mais compatível)
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

    def modify_sl(self, ticket, new_sl):
        """Modifica o Stop Loss de uma posição aberta para Trail/Shield."""
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": self.symbol,
            "sl": float(new_sl),
            "position": ticket,
        }
        result = mt5.order_send(request)
        if result is None:
            logger.error(f"[FAIL] Erro crítico no modify_sl: {mt5.last_error()}")
            return False
            
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.warning(f"[WARN] Falha ao modificar SL: {result.retcode}")
            return False
        return True

    def close_all_positions(self):
        """Fecha todas as posições abertas pelo robô (Magic Number)."""
        positions = mt5.positions_get(magic=self.magic)
        if positions:
            for pos in positions:
                # The following block seems to be misplaced logic intended for an AssetWorker or similar class
                # and contains undefined variables (self.trailing_activated, current_profit_points, risk_dist,
                # LogColors, self.entry_price, tick_size).
                # It also has a syntax error with dictionary key-value pairs outside a dictionary.
                # I am commenting it out as it cannot be integrated into ExecutionEngine.close_all_positions
                # without significant refactoring and context from other classes.
                #
                # if not self.trailing_activated and current_profit_points >= risk_dist:
                #   logger.info(f"🛡️ {LogColors.CYAN}[SHIELD ACTIVATED]{LogColors.RESET} {self.symbol} | Preço atingiu 1x Risco. Protegendo no BE.")
                #   new_sl = self.entry_price + (tick_size if pos.type == mt5.ORDER_TYPE_BUY else -tick_size) # BE + 1 tick
                #   new_sl = self.engine.normalize_price(new_sl)
                #   if self.engine.modify_sl(pos.ticket, new_sl):
                #       self.trailing_activated = True
                #   "action": mt5.TRADE_ACTION_DEAL, # This line and subsequent lines are syntax errors here
                #   "symbol": pos.symbol,
                #   "pos": pos.ticket,
                #   "volume": pos.volume,
                #   "type": type,

                tick = mt5.symbol_info_tick(pos.symbol)
                type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
                price = tick.bid if type == mt5.ORDER_TYPE_SELL else tick.ask
                price = self.normalize_price(price)
                
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": pos.symbol,
                    "position": pos.ticket, # Changed 'pos' to 'position' as per MT5 API for closing by ticket
                    "volume": pos.volume,
                    "type": type,
                    "price": float(price),
                    "magic": self.magic,
                    "comment": "Quant - Close All",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }
                mt5.order_send(request)
            logger.info(f"🧹 {len(positions)} posições fechadas.")

    def shutdown(self):
        """Finaliza a conexão com o MT5."""
        mt5.shutdown()
        logger.info("[MT5] Conexão MT5 encerrada.")
