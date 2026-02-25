
import sys
print("Start", flush=True)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
print("Env set", flush=True)
import pandas as pd
print("Pandas imported", flush=True)
import numpy as np
print("Numpy imported", flush=True)
