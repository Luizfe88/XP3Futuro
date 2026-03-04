"""
Resolver Contratos Futuros MT5 - Mês Atual
==========================================
Converte os códigos genéricos (ex: WIN$N) para os contratos
do mês atual vigente na B3 (ex: WINJ26).
"""

import re
from datetime import date

# ── Caminho do terminal MT5 ───────────────────────────────────────────────────
MT5_PATH = r"C:\MetaTrader 5 Terminal\terminal64.exe"

# ── Mapeamento mês → letra da B3 ─────────────────────────────────────────────
MONTH_CODES = {
    1: "F", 2: "G",  3: "H",  4: "J",
    5: "K", 6: "M",  7: "N",  8: "Q",
    9: "U", 10: "V", 11: "X", 12: "Z",
}

VALID_MONTHS = {
    "WIN": [2, 4, 6, 8, 10, 12],
    "IND": [2, 4, 6, 8, 10, 12],
    "WSP": [2, 4, 6, 8, 10, 12],
    "WDO": list(range(1, 13)),
    "DOL": list(range(1, 13)),
    "CCM": [3, 5, 7, 9, 12],
    "BGI": [1, 3, 5, 7, 9, 11],
    "ICF": [3, 5, 7, 9, 12],
    "BIT": list(range(1, 13)),
    "DI1": list(range(1, 13)),
}

GENERIC_CONTRACTS = [
    "WIN$N", "IND$N", "WSP$N", "WDO$N", "DOL$N",
    "CCM$N", "BGI$N", "ICF$N", "BIT$N", "DI1$N",
]

B3_FUTURE_RE = re.compile(
    r'^([A-Z0-9]{2,4})'
    r'(\$N|[FGHJKMNQUVXZ]\d{2})'
    r'([A-Z0-9@._-]*)$',
    re.IGNORECASE
)


# ── Resolução de contratos ────────────────────────────────────────────────────

def next_valid_month(base_code, ref_date):
    valid = VALID_MONTHS.get(base_code, list(range(1, 13)))
    month, year = ref_date.month, ref_date.year
    for _ in range(13):
        if month in valid:
            return month, year
        month += 1
        if month > 12:
            month, year = 1, year + 1
    raise ValueError(f"Nenhum mês válido para {base_code}")


def resolve_contract(generic, ref_date=None):
    if ref_date is None:
        ref_date = date.today()
    if "$" not in generic:
        return generic
    base_code = generic.split("$")[0]
    month, year = next_valid_month(base_code, ref_date)
    return f"{base_code}{MONTH_CODES[month]}{str(year)[-2:]}"


def resolve_all(contracts=None, ref_date=None):
    if contracts is None:
        contracts = GENERIC_CONTRACTS
    if ref_date is None:
        ref_date = date.today()
    return {c: resolve_contract(c, ref_date) for c in contracts}


def get_prefix(symbol):
    m = re.match(r'^([A-Z0-9]{2,4}?)([FGHJKMNQUVXZ]\d{2}|\$N)', symbol, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return symbol[:3].upper()


# ── Helpers MT5 ───────────────────────────────────────────────────────────────

def is_b3_future(symbol_name, expected_prefix):
    m = B3_FUTURE_RE.match(symbol_name)
    if not m:
        return False
    if m.group(1).upper() != expected_prefix.upper():
        return False
    if "." in m.group(3):
        return False
    return True


def find_b3_symbol(mt5, expected_symbol):
    prefix = get_prefix(expected_symbol)
    all_symbols = mt5.symbols_get() or []

    # Match exato ou com sufixo de corretora
    for s in all_symbols:
        if s.name.upper() == expected_symbol.upper():
            return s.name
        if s.name.upper().startswith(expected_symbol.upper()) and is_b3_future(s.name, prefix):
            return s.name

    # Formato genérico $N
    generic = prefix + "$N"
    for s in all_symbols:
        if s.name.upper() == generic.upper():
            return s.name

    return None


# ── Busca de cotações no MT5 ──────────────────────────────────────────────────

def get_mt5_symbols(resolved):
    import MetaTrader5 as mt5

    if not mt5.initialize(path=MT5_PATH):
        print(f"  ⚠  Falha ao inicializar MT5: {mt5.last_error()}")
        return {}

    results = {}
    for generic, symbol in resolved.items():
        prefix = get_prefix(symbol)

        candidates = list(dict.fromkeys([
            symbol,
            symbol + "F",
            symbol + "N",
            prefix + "$N",
        ]))

        found_sym = None
        found_info = None

        for candidate in candidates:
            mt5.symbol_select(candidate, True)
            info = mt5.symbol_info(candidate)
            if info:
                found_sym, found_info = candidate, info
                break

        if not found_info:
            found_sym = find_b3_symbol(mt5, symbol)
            if found_sym:
                mt5.symbol_select(found_sym, True)
                found_info = mt5.symbol_info(found_sym)

        if found_info:
            tick = mt5.symbol_info_tick(found_sym)
            results[generic] = {
                "symbol":      found_sym,
                "resolved":    symbol,
                "description": found_info.description,
                "bid":         tick.bid   if tick else None,
                "ask":         tick.ask   if tick else None,
                "last":        tick.last  if tick else None,
                "spread":      found_info.spread,
                "digits":      found_info.digits,
            }
        else:
            results[generic] = {
                "symbol":   symbol,
                "resolved": symbol,
                "error":    f"Não encontrado (tentativas: {', '.join(candidates)})",
            }

    mt5.shutdown()
    return results


# ── Diagnóstico ───────────────────────────────────────────────────────────────

def diagnostico_mt5(resolved):
    import MetaTrader5 as mt5

    if not mt5.initialize(path=MT5_PATH):
        return

    prefixes = sorted({get_prefix(v) for v in resolved.values()})
    all_syms = mt5.symbols_get() or []

    print(f"\n  {'─'*54}")
    print(f"  DIAGNÓSTICO – Símbolos encontrados no MT5:\n")
    for pfx in prefixes:
        matches = [s.name for s in all_syms if s.name.upper().startswith(pfx.upper())]
        label = ", ".join(matches[:8]) if matches else "(nenhum)"
        print(f"  {pfx:>5}: {label}")
    print(f"  {'─'*54}\n")

    mt5.shutdown()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ref_date = date.today()
    print(f"\n{'='*58}")
    print(f"  Resolução de Contratos Futuros B3 – MT5")
    print(f"  Data de referência : {ref_date.strftime('%d/%m/%Y')}")
    print(f"{'='*58}\n")

    resolved = resolve_all(ref_date=ref_date)

    print(f"  {'Genérico':<10}  →  {'Contrato Resolvido'}")
    print(f"  {'-'*10}     {'-'*20}")
    for generic, symbol in resolved.items():
        print(f"  {generic:<10}  →  {symbol}")

    print(f"\n{'='*58}")
    print("  Buscando cotações no MetaTrader 5…")
    print(f"{'='*58}\n")

    mt5_data = get_mt5_symbols(resolved)

    if not mt5_data:
        print("  (MT5 não disponível)\n")
        return resolved

    any_found = False
    for generic, data in mt5_data.items():
        sym_display = data["symbol"]
        extra = f"  ← de {data['resolved']}" if data["symbol"] != data["resolved"] else ""
        print(f"  [{sym_display}]{extra}")
        if "error" in data:
            print(f"    ⚠  {data['error']}")
        else:
            any_found = True
            print(f"    Descrição : {data.get('description', '-')}")
            print(f"    Bid       : {data.get('bid')}")
            print(f"    Ask       : {data.get('ask')}")
            print(f"    Last      : {data.get('last')}")
            print(f"    Spread    : {data.get('spread')}")
        print()

    if not any_found:
        diagnostico_mt5(resolved)

    return resolved


if __name__ == "__main__":
    main()