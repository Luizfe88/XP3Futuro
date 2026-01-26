import ast
from pathlib import Path

BOT_FILE = Path("bot.py")
UTILS_FILE = Path("utils.py")

def get_defined_functions(tree):
    return {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}

def get_imported_utils_functions(tree):
    funcs = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "utils":
            for n in node.names:
                funcs.add(n.name)
    return funcs

def get_utils_calls(tree):
    calls = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Name) and node.func.value.id == "utils":
                    calls.add(node.func.attr)
    return calls

def main():
    bot_tree = ast.parse(BOT_FILE.read_text(encoding="utf-8"))
    utils_tree = ast.parse(UTILS_FILE.read_text(encoding="utf-8"))

    utils_funcs = get_defined_functions(utils_tree)
    imported_funcs = get_imported_utils_functions(bot_tree)
    called_funcs = get_utils_calls(bot_tree)

    print("\n🧪 VALIDANDO CHAMADAS utils.py ↔ bot.py\n")

    missing_imports = imported_funcs - utils_funcs
    missing_calls = called_funcs - utils_funcs

    if not missing_imports and not missing_calls:
        print("✅ Nenhum problema encontrado!")
        return

    if missing_imports:
        print("❌ Imports inválidos:")
        for f in sorted(missing_imports):
            print(f"   - {f}")

    if missing_calls:
        print("\n❌ Chamadas inexistentes em utils.py:")
        for f in sorted(missing_calls):
            print(f"   - utils.{f}()")

if __name__ == "__main__":
    main()