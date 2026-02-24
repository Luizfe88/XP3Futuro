import sys
import os

print("--- Verificando Imports ---")

def show_origin(mod_name, mod_obj):
    try:
        path = getattr(mod_obj, "__file__", None)
        print(f"   • origem: {path}")
        if path and ("xp3v5" in path.lower()):
            print(f"   ❌ ALERTA: {mod_name} carregado de xp3v5 (deve vir de xp3future)")
        elif path:
            print(f"   ✅ OK: {mod_name} carregado de xp3future")
    except Exception as e:
        print(f"   ⚠️ Não foi possível obter origem de {mod_name}: {e}")

try:
    print("Importando xp3future como config...")
    import xp3future as config
    print("✅ config importado")
    show_origin("config", config)
except ImportError as e:
    print(f"❌ Erro ao importar config: {e}")
except Exception as e:
    print(f"❌ Erro genérico em config: {e}")

try:
    print("Importando utils...")
    import utils
    print("✅ utils importado")
    show_origin("utils", utils)
except ImportError as e:
    print(f"❌ Erro ao importar utils: {e}")
except Exception as e:
    print(f"❌ Erro genérico em utils: {e}")

try:
    print("Importando validation...")
    import validation
    print("✅ validation importado")
    show_origin("validation", validation)
except ImportError as e:
    print(f"❌ Erro ao importar validation: {e}")
except Exception as e:
    print(f"❌ Erro genérico em validation: {e}")

try:
    print("Importando botfuturo...")
    import botfuturo
    print("✅ botfuturo importado (nota: execução não iniciada)")
    show_origin("botfuturo", botfuturo)
    # Verifica origem de 'bot' quando importado por botfuturo
    try:
        import importlib
        bot = importlib.import_module("bot")
        print("   ↪ import bot via importlib")
        show_origin("bot", bot)
    except Exception as e:
        print(f"   ⚠️ Não foi possível importar/verificar bot: {e}")
except ImportError as e:
    print(f"❌ Erro ao importar botfuturo: {e}")
except Exception as e:
    print(f"❌ Erro genérico em botfuturo: {e}")

try:
    print("Importando otimizador_semanal...")
    import otimizador_semanal
    print("✅ otimizador_semanal importado")
    show_origin("otimizador_semanal", otimizador_semanal)
except ImportError as e:
    print(f"❌ Erro ao importar otimizador_semanal: {e}")
except Exception as e:
    print(f"❌ Erro genérico em otimizador_semanal: {e}")

try:
    print("Importando dashboard...")
    # Dashboard geralmente roda com streamlit run, mas o import deve passar
    import dashboard
    print("✅ dashboard importado")
    show_origin("dashboard", dashboard)
except ImportError as e:
    print(f"❌ Erro ao importar dashboard: {e}")
except Exception as e:
    print(f"❌ Erro genérico em dashboard: {e}")

print("--- Verificação Concluída ---")
