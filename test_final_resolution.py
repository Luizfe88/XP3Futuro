from utils import calculate_current_b3_contract
from datetime import date

assets = ['WIN', 'IND', 'WSP', 'WDO', 'DOL', 'CCM', 'BGI', 'ICF', 'BIT', 'DI1']
print(f"--- Verificando Resolução Matemática (Data: {date.today()}) ---")
for a in assets:
    res = calculate_current_b3_contract(a)
    print(f"{a:<10} -> {res}")
