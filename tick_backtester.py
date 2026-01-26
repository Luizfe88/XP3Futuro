import sqlite3

class TickRecorder:
    """Grava ticks para posterior análise"""
    
    def __init__(self, db_path="ticks.db"):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS ticks (
                timestamp INTEGER,
                symbol TEXT,
                bid REAL,
                ask REAL,
                volume INTEGER
            )
        """)
    
    def record_tick(self, symbol):
        tick = mt5.symbol_info_tick(symbol)
        if tick:
            self.conn.execute(
                "INSERT INTO ticks VALUES (?, ?, ?, ?, ?)",
                (tick.time, symbol, tick.bid, tick.ask, tick.volume)
            )
            self.conn.commit()
    
    def replay_backtest(self, strategy, start_date, end_date):
        """
        Executa backtest usando ticks gravados.
        Simula ordens IOC com slippage real observado.
        """
        cursor = self.conn.execute(
            "SELECT * FROM ticks WHERE timestamp BETWEEN ? AND ?",
            (start_date, end_date)
        )
        
        for row in cursor:
            timestamp, symbol, bid, ask, volume = row
            # Simula estratégia tick a tick
            strategy.process_tick(timestamp, symbol, bid, ask, volume)
        
        return strategy.get_results()