"""
Teste de resolução de contratos $N
Verifica se WIN$N -> WINM26 (ou similar) corretamente
"""
import MetaTrader5 as mt5
import sys

def main():
    print("🔌 Conectando ao MT5...")
    if not mt5.initialize():
        print(f"❌ Falha ao conectar MT5: {mt5.last_error()}")
        sys.exit(1)
    print("✅ MT5 conectado.\n")

    import utils
    import config_futures

    patterns = list(config_futures.FUTURES_CONFIGS.keys())
    print(f"📋 Testando {len(patterns)} padrões: {patterns}\n")
    print("-" * 60)

    sucesso = 0
    falha = 0

    for pattern in patterns:
        # Extração da base (lógica corrigida)
        pat = pattern.upper().strip()
        if "$" in pat:
            pat = pat.split("$")[0]
        base = "".join([c for c in pat if c.isalpha()])

        # Busca candidatos
        candidates = utils.get_futures_candidates(base)
        if candidates:
            best = candidates[0]
            sym = best.get("symbol", "?")
            days = best.get("days_to_exp", "?")
            vol  = best.get("volume", 0)
            print(f"  ✅ {pattern:12} -> base={base:5} -> {sym:12} (vence em {days} dias, vol={vol:.0f})")
            sucesso += 1
        else:
            print(f"  ❌ {pattern:12} -> base={base:5} -> NENHUM CANDIDATO ENCONTRADO")
            falha += 1

    print("-" * 60)
    print(f"\n📊 Resultado: {sucesso} resolvidos | {falha} falhos")

    # Testa resolve_symbol para os padrões
    print("\n🔁 Testando resolve_symbol():")
    print("-" * 60)
    for pattern in patterns:
        resolved = utils.resolve_symbol(pattern)
        status = "✅" if resolved and resolved != pattern else "⚠️ "
        print(f"  {status} resolve_symbol({pattern}) -> {resolved}")

    mt5.shutdown()
    print("\n🔌 MT5 desconectado.")

if __name__ == "__main__":
    main()
