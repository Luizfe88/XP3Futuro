from daily_analysis_logger import daily_logger
import os
from pathlib import Path

print("--- Teste de Log Dinâmico (Transparency Enhancement) ---")

# Limpa logs anteriores para o teste ser limpo (opcional)
log_file = daily_logger._get_log_filename()
if log_file.exists():
    log_file.unlink()

test_cases = [
    {
        "name": "RSI Esticado",
        "symbol": "EURUSD",
        "score": 65,
        "indicators": {"rsi": 75, "adx": 30, "volume_ratio": 1.2, "score_log": {}},
        "expected": "Aguardando correção (RSI Esticado)"
    },
    {
        "name": "ADX Baixo",
        "symbol": "GBPUSD",
        "score": 62,
        "indicators": {"rsi": 50, "adx": 15, "volume_ratio": 1.1, "score_log": {}},
        "expected": "Sem força de tendência (ADX Baixo)"
    },
    {
        "name": "Volume Insuficiente",
        "symbol": "USDJPY",
        "score": 68,
        "indicators": {"rsi": 45, "adx": 35, "volume_ratio": 0.5, "score_log": {}},
        "expected": "Volume institucional insuficiente"
    },
    {
        "name": "Score Baixo - Penalidade ADX",
        "symbol": "AUDUSD",
        "score": 45,
        "indicators": {
            "rsi": 50, "adx": 12, "volume_ratio": 1.0,
            "score_log": {"PENALTY_NO_TREND": -20}
        },
        "expected": "Configuração de Risco: Sem tendência clara"
    },
    {
        "name": "Score Baixo - Falta Confirmação",
        "symbol": "USDCAD",
        "score": 40,
        "indicators": {
            "rsi": 55, "adx": 25, "volume_ratio": 0.9,
            "score_log": {"BASE": 20, "MOMENTUM": 15}
        },
        "expected": "Configuração de Risco: Falta cruzamento MACD e Volume abaixo do ideal"
    }
]

for case in test_cases:
    print(f"Testando: {case['name']}...")
    daily_logger.log_analysis(
        symbol=case['symbol'],
        signal="BUY",
        strategy="TREND",
        score=case['score'],
        rejected=True,
        reason="Rejeição genérica", # Deve ser substituído pela lógica dinâmica
        indicators=case['indicators']
    )

print(f"\nLog gravado em: {daily_logger.current_file}")

if daily_logger.current_file and daily_logger.current_file.exists():
    with open(daily_logger.current_file, 'r', encoding='utf-8') as f:
        content = f.read()
        print("\n--- Conteúdo do Log ---")
        print(content)
        
        all_passed = True
        for case in test_cases:
            if case['expected'] in content:
                print(f"✅ PASSED: '{case['expected']}' encontrado para {case['name']}")
            else:
                print(f"❌ FAILED: '{case['expected']}' NÃO encontrado para {case['name']}")
                all_passed = False
        
        if all_passed:
            print("\n🎉 TODOS OS TESTES PASSARAM!")
        else:
            print("\n⚠️ ALGUNS TESTES FALHARAM.")
else:
    print("❌ Erro: Arquivo de log não encontrado.")
