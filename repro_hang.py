
import sys
print("DEBUG: Init", flush=True)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import json
print("DEBUG: json imported", flush=True)

# Teste 1: Importar Pandas direto (como no original)
print("DEBUG: importing pandas...", flush=True)
import pandas as pd
print("DEBUG: pandas imported", flush=True)

print("DEBUG: importing numpy...", flush=True)
import numpy as np
print("DEBUG: numpy imported", flush=True)

import time
import threading
import logging
print("DEBUG: logging imported", flush=True)
import asyncio
print("DEBUG: asyncio imported", flush=True)
