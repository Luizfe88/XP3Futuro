import sys
import os

print(f"🚀 Iniciando XP3 Future via {__file__}...", flush=True)

try:
    # Importa o módulo bot.py (onde está a lógica principal)
    # Não alteramos sys.path para evitar conflitos com libs padrão
    import bot
    
    # Executa a função main() ou o código de inicialização do bot
    # Prioritiza a execução do main() do bot.py que gerencia todas as threads
    if hasattr(bot, 'main'):
        print("✅ Ponto de entrada main() encontrado. Iniciando sistema completo...", flush=True)
        bot.main()
    elif hasattr(bot, 'fast_loop'):
        # Fallback para versão sem main()
        print("✅ Módulo bot importado. Iniciando modo fast_loop...", flush=True)
        
        if hasattr(bot, 'setup_logging'):
            bot.setup_logging()
            
        if hasattr(bot, 'validate_futures_only_mode'):
            if not bot.validate_futures_only_mode():
                sys.exit(1)
                
        if hasattr(bot, 'load_optimized_params'):
            bot.load_optimized_params()
            
        bot.fast_loop()
    else:
        print("❌ Erro: Não foi possível encontrar o ponto de entrada no bot.py")
        
except KeyboardInterrupt:
    print("\n🛑 Interrompido pelo usuário.")
except Exception as e:
    print(f"❌ Erro fatal ao iniciar: {e}")
    import traceback
    traceback.print_exc()
