import os
import functools
print = functools.partial(print, flush=True)
os.environ["MKL_NUM_THREADS"] = "1"
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("BLIS_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")
os.environ.setdefault("KMP_BLOCKTIME", "0")
os.environ.setdefault("KMP_AFFINITY", "disabled")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")
os.environ.setdefault("KMP_WARNINGS", "0")
print("Tentando carregar Sklearn Ensemble...")
try:
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as TErr
    def _imp():
        from sklearn.ensemble import RandomForestClassifier
        return True
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_imp)
        try:
            ok = fut.result(timeout=10)
            print("Sucesso!" if ok else "Falhou")
        except TErr:
            print("Timeout ao importar sklearn.ensemble (possível conflito de threads/DLL)")
        except Exception as e:
            print(f"Falhou: {e}")
except Exception as e:
    print(f"Infra de import com timeout falhou: {e}")
