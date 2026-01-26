import os
import logging
from datetime import datetime
from pathlib import Path

# Configuração de diretório de logs de rejeição
REJECTION_LOGS_DIR = Path("logs/rejections")
REJECTION_LOGS_DIR.mkdir(parents=True, exist_ok=True)

class RejectionLogger:
    """
    Sistema de log especializado para registrar e explicar
    por que uma oportunidade de trade foi rejeitada pelos filtros.
    Gera um arquivo por dia.
    """
    
    def __init__(self):
        self._current_date = datetime.now().strftime("%Y-%m-%d")
        self._logger = self._setup_logger()

    def _setup_logger(self):
        log_file = REJECTION_LOGS_DIR / f"rejections_{self._current_date}.log"
        
        logger = logging.getLogger(f"rejection_logger_{self._current_date}")
        logger.setLevel(logging.INFO)
        
        # Evita duplicar handlers se o logger já existir
        if not logger.handlers:
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
        return logger

    def log_rejection(self, symbol: str, strategy: str, reason: str, details: dict = None):
        """
        Registra uma rejeição de trade.
        
        Args:
            symbol: Ativo (ex: PETR4)
            strategy: Nome da estratégia ou filtro (ex: VolumeFilter, RSI)
            reason: Motivo curto da rejeição
            details: Dicionário com dados técnicos (ex: {'vol_ratio': 1.2, 'threshold': 1.5})
        """
        # Checa se o dia mudou para rotacionar o log
        now_date = datetime.now().strftime("%Y-%m-%d")
        if now_date != self._current_date:
            self._current_date = now_date
            self._logger = self._setup_logger()

        detail_str = f" | Details: {details}" if details else ""
        message = f"❌ [{symbol}] [{strategy}] {reason}{detail_str}"
        self._logger.info(message)

# Instância global
rejection_logger = RejectionLogger()

def log_trade_rejection(symbol: str, strategy: str, reason: str, details: dict = None):
    """Função utilitária para acesso rápido"""
    rejection_logger.log_rejection(symbol, strategy, reason, details)
