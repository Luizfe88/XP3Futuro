
import sys
import os
import faulthandler

faulthandler.enable()

print(f"Python version: {sys.version}", flush=True)
print(f"Executable: {sys.executable}", flush=True)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

print("Importing numpy...", flush=True)
try:
    import numpy
    print(f"Numpy version: {numpy.__version__}", flush=True)
except ImportError as e:
    print(f"Numpy import failed: {e}", flush=True)

print("Importing pandas...", flush=True)
try:
    # Tenta importar com verbose para ver onde trava
    import pandas as pd
    print(f"Pandas version: {pd.__version__}", flush=True)
except ImportError as e:
    print(f"Pandas import failed: {e}", flush=True)
except Exception as e:
    print(f"Pandas unknown error: {e}", flush=True)

print("Done.", flush=True)
