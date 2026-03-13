import sys
import io
import logging

def test_unicode_logging():
    # Force UTF-8 for standard output/error to handle emojis on Windows
    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        handlers=[
            logging.FileHandler("tmp_unicode_test.log", encoding="utf-8"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger("TestUnicode")
    
    try:
        logger.info("Testando emoji: ✅ ⚠️ 🚀 📉 📈")
        print("Console: ✅ ⚠️ 🚀 📉 📈")
        print("Teste concluído com sucesso!")
    except UnicodeEncodeError as e:
        print(f"Erro de codificação: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Erro inesperado: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_unicode_logging()
