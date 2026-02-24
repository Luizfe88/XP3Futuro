import sqlite3
import os
from datetime import datetime, timedelta
import pandas as pd
import logging
import threading

logger = logging.getLogger("database")

DB_PATH = "xp3_trades.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            volume REAL NOT NULL,
            entry_price REAL NOT NULL,
            exit_price REAL,
            sl REAL,
            tp REAL,
            pnl_money REAL,
            pnl_pct REAL,
            reason TEXT,
            ml_reward REAL,
            -- Novas colunas para análise avançada
            strategy TEXT DEFAULT 'ELITE',
            ml_confidence REAL DEFAULT 0.0,
            ml_prediction TEXT DEFAULT '',
            atr_pct REAL DEFAULT 0.0,
            vix_level REAL DEFAULT 0.0,
            order_flow_delta REAL DEFAULT 0.0,
            duration_minutes INTEGER DEFAULT 0,
            exit_time DATETIME,
            ab_group TEXT DEFAULT 'A'
        )
    ''')
    conn.commit()
    conn.close()

class StateManager:
    def __init__(self, db_path: str = "trading_state.db"):
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.lock = os.name and threading.RLock() if hasattr(__import__('threading'), 'RLock') else None
        self._init_schema()
    def _init_schema(self):
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS daily_state (
                trading_date DATE PRIMARY KEY,
                equity_start REAL NOT NULL,
                equity_max REAL NOT NULL,
                trades_count INTEGER DEFAULT 0,
                wins_count INTEGER DEFAULT 0,
                loss_streak INTEGER DEFAULT 0,
                circuit_breaker_active BOOLEAN DEFAULT 0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                CHECK (equity_start > 0),
                CHECK (equity_max >= equity_start),
                CHECK (trades_count >= 0),
                CHECK (wins_count >= 0),
                CHECK (loss_streak >= 0)
            )
        """)
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS symbol_limits (
                symbol TEXT NOT NULL,
                trading_date DATE NOT NULL,
                trades_count INTEGER DEFAULT 0,
                losses_count INTEGER DEFAULT 0,
                last_sl_time TIMESTAMP,
                cooldown_until TIMESTAMP,
                PRIMARY KEY (symbol, trading_date),
                CHECK (trades_count >= losses_count)
            )
        """)
        self.db.commit()
    def save_state_atomic(self, state: dict):
        cur = self.db.cursor()
        try:
            self.db.execute("BEGIN TRANSACTION")
            cur.execute("""
                INSERT OR REPLACE INTO daily_state 
                (trading_date, equity_start, equity_max, trades_count, wins_count, loss_streak, circuit_breaker_active, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (
                state.get("trading_date"),
                float(state.get("equity_start", 0) or 0),
                float(state.get("equity_max", 0) or 0),
                int(state.get("trades_count", 0) or 0),
                int(state.get("wins_count", 0) or 0),
                int(state.get("loss_streak", 0) or 0),
                int(bool(state.get("circuit_breaker_active", False))),
            ))
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise
    def get_today_state(self) -> dict:
        today = datetime.now().date().isoformat()
        df = pd.read_sql_query("SELECT * FROM daily_state WHERE trading_date = date(?)", self.db, params=(today,))
        if len(df) == 0:
            return {}
        row = df.iloc[0].to_dict()
        return row
    def reset_daily_if_needed(self):
        today = datetime.now().date().isoformat()
        df = pd.read_sql_query("SELECT * FROM daily_state WHERE trading_date = date(?)", self.db, params=(today,))
        if len(df) == 0:
            self.save_state_atomic({
                "trading_date": today,
                "equity_start": 1.0,
                "equity_max": 1.0,
                "trades_count": 0,
                "wins_count": 0,
                "loss_streak": 0,
                "circuit_breaker_active": False,
            })
    def update_symbol_limits(self, symbol: str, trades_delta: int = 0, losses_delta: int = 0):
        today = datetime.now().date().isoformat()
        cur = self.db.cursor()
        cur.execute("""
            INSERT INTO symbol_limits (symbol, trading_date, trades_count, losses_count)
            VALUES (?, date(?), 0, 0)
            ON CONFLICT(symbol, trading_date) DO NOTHING
        """, (symbol, today))
        cur.execute("""
            UPDATE symbol_limits
            SET trades_count = trades_count + ?, losses_count = losses_count + ?, last_sl_time = CASE WHEN ? > 0 THEN CURRENT_TIMESTAMP ELSE last_sl_time END
            WHERE symbol = ? AND trading_date = date(?)
        """, (int(trades_delta), int(losses_delta), int(losses_delta), symbol, today))
        self.db.commit()
    def get_symbol_limits(self, symbol: str) -> dict:
        today = datetime.now().date().isoformat()
        df = pd.read_sql_query("SELECT * FROM symbol_limits WHERE symbol = ? AND trading_date = date(?)", self.db, params=(symbol, today))
        if len(df) == 0:
            return {"trades_count": 0, "losses_count": 0}
        return df.iloc[0].to_dict()

def migrate_db():
    """Adiciona novas colunas se não existirem (migrate schema)."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    new_columns = [
        ("strategy", "TEXT DEFAULT 'ELITE'"),
        ("ml_confidence", "REAL DEFAULT 0.0"),
        ("ml_prediction", "TEXT DEFAULT ''"),
        ("atr_pct", "REAL DEFAULT 0.0"),
        ("vix_level", "REAL DEFAULT 0.0"),
        ("order_flow_delta", "REAL DEFAULT 0.0"),
        ("duration_minutes", "INTEGER DEFAULT 0"),
        ("exit_time", "DATETIME"),
        ("ab_group", "TEXT DEFAULT 'A'")
    ]
    
    for col_name, col_def in new_columns:
        try:
            cursor.execute(f"ALTER TABLE trades ADD COLUMN {col_name} {col_def}")
            logger.info(f"✅ Coluna {col_name} adicionada")
        except sqlite3.OperationalError:
            pass  # Coluna já existe
    
    conn.commit()
    conn.close()

def save_trade(
    symbol, side, volume, entry_price, exit_price=None,
    sl=None, tp=None, pnl_money=0.0, pnl_pct=0.0,
    reason="", ml_reward=0.0, strategy="ELITE",
    ml_confidence=0.0, ml_prediction="", atr_pct=0.0,
    vix_level=0.0, order_flow_delta=0.0, duration_minutes=0,
    ab_group="A"
):
    init_db()
    migrate_db()
    
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''
        INSERT INTO trades 
        (symbol, side, volume, entry_price, exit_price, sl, tp, pnl_money, pnl_pct, 
         reason, ml_reward, strategy, ml_confidence, ml_prediction, atr_pct,
         vix_level, order_flow_delta, duration_minutes, exit_time, ab_group)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (symbol, side, volume, entry_price, exit_price, sl, tp, pnl_money, pnl_pct,
          reason, ml_reward, strategy, ml_confidence, ml_prediction, atr_pct,
          vix_level, order_flow_delta, duration_minutes, datetime.now() if exit_price else None, ab_group))
    conn.commit()
    conn.close()

def get_trades_by_date(target_date_str: str):
    """Busca todos os trades de uma data específica (formato 'YYYY-MM-DD')."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT * FROM trades WHERE date(timestamp) = date(?)"
    df = pd.read_sql_query(query, conn, params=(target_date_str,))
    conn.close()
    return df

def get_win_rate_report(lookback_days: int = 30) -> dict:
    """
    Gera relatório de win rate geral e por estratégia.
    """
    init_db()
    migrate_db()
    conn = sqlite3.connect(DB_PATH)
    cutoff = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    
    # Win rate geral
    df = pd.read_sql_query(f"""
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN pnl_money > 0 THEN 1 ELSE 0 END) as wins,
            SUM(pnl_money) as total_pnl,
            AVG(pnl_pct) as avg_pnl_pct
        FROM trades
        WHERE date(timestamp) >= date('{cutoff}') AND exit_price IS NOT NULL
    """, conn)
    
    # Win rate por estratégia
    df_strategy = pd.read_sql_query(f"""
        SELECT 
            strategy,
            COUNT(*) as total,
            SUM(CASE WHEN pnl_money > 0 THEN 1 ELSE 0 END) as wins,
            AVG(ml_confidence) as avg_ml_conf
        FROM trades
        WHERE date(timestamp) >= date('{cutoff}') AND exit_price IS NOT NULL
        GROUP BY strategy
    """, conn)
    
    # Win rate por grupo AB
    df_ab = pd.read_sql_query(f"""
        SELECT 
            ab_group,
            COUNT(*) as total,
            SUM(CASE WHEN pnl_money > 0 THEN 1 ELSE 0 END) as wins
        FROM trades
        WHERE date(timestamp) >= date('{cutoff}') AND exit_price IS NOT NULL
        GROUP BY ab_group
    """, conn)
    
    conn.close()
    
    total = df['total'].iloc[0] if len(df) > 0 else 0
    wins = df['wins'].iloc[0] if len(df) > 0 else 0
    total = int(total or 0)
    wins = int(wins or 0)
    
    result = {
        "period_days": lookback_days,
        "total_trades": total,
        "wins": wins,
        "losses": int(total - wins),
        "win_rate": (wins / total * 100) if total > 0 else 0,
        "total_pnl": float(df['total_pnl'].iloc[0] or 0),
        "avg_pnl_pct": float(df['avg_pnl_pct'].iloc[0] or 0),
        "by_strategy": {},
        "by_ab_group": {}
    }
    
    for _, row in df_strategy.iterrows():
        strat = row['strategy'] or 'UNKNOWN'
        wr = (row['wins'] / row['total'] * 100) if row['total'] > 0 else 0
        result["by_strategy"][strat] = {
            "total": int(row['total']),
            "wins": int(row['wins']),
            "win_rate": wr,
            "avg_ml_conf": float(row['avg_ml_conf'] or 0)
        }
    
    for _, row in df_ab.iterrows():
        grp = row['ab_group'] or 'A'
        wr = (row['wins'] / row['total'] * 100) if row['total'] > 0 else 0
        result["by_ab_group"][grp] = {
            "total": int(row['total']),
            "wins": int(row['wins']),
            "win_rate": wr
        }
    
    return result

def get_symbol_statistics(symbol: str, lookback_days: int = 30) -> dict:
    """📊 Estatísticas reais do símbolo."""
    try:
        init_db()
        migrate_db()
        conn = sqlite3.connect(DB_PATH)
        cutoff = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        
        df = pd.read_sql_query(f"""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN pnl_money > 0 THEN 1 ELSE 0 END) as wins,
                AVG(ABS(tp - entry_price) / NULLIF(ABS(sl - entry_price), 0)) as avg_rr
            FROM trades
            WHERE symbol = ? AND date(timestamp) >= date('{cutoff}') AND exit_price IS NOT NULL
        """, conn, params=(symbol,))
        conn.close()
        
        if len(df) > 0 and df['total'].iloc[0] >= 10:
            total = df['total'].iloc[0]
            wins = df['wins'].iloc[0]
            return {
                "win_rate": wins / total if total > 0 else 0.55,
                "avg_rr": float(df['avg_rr'].iloc[0] or 2.0),
                "total_trades": int(total),
                "last_updated": datetime.now()
            }
        
        return {"win_rate": 0.55, "avg_rr": 2.0, "total_trades": 0}
    
    except Exception as e:
        logger.error(f"Erro stats {symbol}: {e}")
        return {"win_rate": 0.55, "avg_rr": 2.0, "total_trades": 0}
