import sys
import os

print(f"🚀 Iniciando XP3 Future via {__file__}...", flush=True)

try:
    # Importa o módulo bot.py (onde está a lógica principal)
    # Não alteramos sys.path para evitar conflitos com libs padrão
    import bot
    
    # Executa a função main() ou o código de inicialização do bot
    if hasattr(bot, 'fast_loop'):
        # Se bot.py não tiver main(), executa o setup e o loop
        print("✅ Módulo bot importado. Iniciando sistema...", flush=True)
        
        # Chama setup se necessário (bot.py executa setup no import, mas fast_loop precisa ser chamado)
        if hasattr(bot, 'setup_logging'):
            bot.setup_logging()
            
        if hasattr(bot, 'validate_futures_only_mode'):
            if not bot.validate_futures_only_mode():
                sys.exit(1)
                
        if hasattr(bot, 'load_optimized_params'):
            bot.load_optimized_params()
            
        if hasattr(bot, 'utils') and hasattr(bot.utils, 'start_watchdog'):
            bot.utils.start_watchdog()
            
        # Inicia o loop principal
        bot.fast_loop()
        
    elif hasattr(bot, 'main'):
        bot.main()
    else:
        print("❌ Erro: Não foi possível encontrar o ponto de entrada no bot.py")
        
except KeyboardInterrupt:
    print("\n🛑 Interrompido pelo usuário.")
except Exception as e:
    print(f"❌ Erro fatal ao iniciar: {e}")
    import traceback
    traceback.print_exc()
