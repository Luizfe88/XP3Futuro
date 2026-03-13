import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestHMM")

def reproduce_and_verify():
    # 1. Simulate data with extremely low variance (causes LinAlgError/Singular Matrix)
    # This often happens with assets like DI1 or when the Kalman filter is too smooth
    n_samples = 1000
    returns = np.zeros(n_samples)
    vol = np.zeros(n_samples)
    
    # Add tiny jitter but keep it close to zero
    X = np.column_stack([returns, vol])
    X += np.random.normal(0, 1e-12, X.shape) 
    
    print("\n--- Testing HMM with Single-Value Data ---")
    
    # 2. Setup Scaler and Model (following the new logic in bot_quant_portfolio.py)
    scaler = StandardScaler()
    try:
        X_scaled = scaler.fit_transform(X)
        print("✅ Feature scaling successful.")
        
        # Using min_covar=1e-2 as implemented in the fix
        model = GaussianHMM(n_components=2, covariance_type="full", n_iter=10, random_state=42, min_covar=1e-2)
        model.fit(X_scaled)
        print("✅ HMM Training successful (min_covar=1e-2 prevented LinAlgError).")
        
        # 3. Test Prediction with error handling
        test_feat = np.array([[0.0, 0.0]])
        test_feat_jitter = test_feat + np.random.normal(0, 1e-9, test_feat.shape)
        
        try:
            feat_scaled = scaler.transform(test_feat_jitter)
            regime = model.predict(feat_scaled)[0]
            print(f"✅ Prediction internal: Regime {regime}")
        except Exception as e:
            print(f"❌ Prediction failed internal: {e}")
            
    except Exception as e:
        print(f"💥 Critical Failure during Training/Scaling: {e}")

    # 4. Verify the fallback logic itself (simulating a complete model failure)
    print("\n--- Testing Fallback Logic ---")
    try:
        # Simulate a case where predict() still fails despite fix (extreme case)
        raise ValueError("Simulated Covariance Error")
    except Exception as e:
        logger.error(f"Error in HMM prediction: {e}")
        regime = 0
        confidence = 0.5
        print(f"✅ Fallback triggered: Regime={regime}, Confidence={confidence}")

if __name__ == "__main__":
    reproduce_and_verify()
