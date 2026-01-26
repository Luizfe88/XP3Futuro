import functools
print = functools.partial(print, flush=True)
print("Tentando importar ML libraries...")
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
try:
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as TErr
    def _imp():
        import ml_optimizer
        return True
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_imp)
        try:
            ok = fut.result(timeout=10)
            print("Sucesso!" if ok else "Falhou")
        except TErr:
            print("Timeout ao importar ml_optimizer (possível conflito de DLL/threads)")
        except Exception as e:
            print(f"Falhou: {e}")
except Exception as e:
    print(f"Infra de import com timeout falhou: {e}")
