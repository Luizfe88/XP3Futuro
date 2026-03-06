import pandas as pd
import logging

# Mock logger
logger = logging.getLogger("test")

def update_correlations_mock(top15_symbols, mock_data):
    """
    Mock of update_correlations to demonstrate the error.
    """
    try:
        data = {}
        for sym in top15_symbols:
            rates = mock_data.get(sym)
            if rates is not None:
                data[sym] = [r["close"] for r in rates]

        if len(data) > 1:
            print(f"Attempting to create DataFrame with keys: {data.keys()}")
            for k, v in data.items():
                print(f"  {k}: length {len(v)}")
            df = pd.DataFrame(data)
            print("✅ Success!")
            return df

    except Exception as e:
        print(f"❌ Error: {e}")
        return None

# Test case 1: Same lengths (Success)
print("--- Test Case 1: Same lengths ---")
mock_data_1 = {
    "SYM1": [{"close": 1.0}] * 50,
    "SYM2": [{"close": 2.0}] * 50
}
update_correlations_mock(["SYM1", "SYM2"], mock_data_1)

# Test case 2: Different lengths (Failure)
print("\n--- Test Case 2: Different lengths ---")
mock_data_2 = {
    "SYM1": [{"close": 1.0}] * 50,
    "SYM2": [{"close": 2.0}] * 49
}
update_correlations_mock(["SYM1", "SYM2"], mock_data_2)

def update_correlations_fixed(top15_symbols, mock_data):
    """
    Fixed version of update_correlations.
    """
    try:
        data = {}
        target_len = 50
        for sym in top15_symbols:
            rates = mock_data.get(sym)
            if rates is not None and len(rates) == target_len:
                data[sym] = [r["close"] for r in rates]
            elif rates is not None:
                print(f"⚠️ Símbolo {sym} retornado com apenas {len(rates)} bars, ignorando na correlação.")

        if len(data) > 1:
            print(f"Attempting to create DataFrame with keys: {data.keys()}")
            for k, v in data.items():
                print(f"  {k}: length {len(v)}")
            df = pd.DataFrame(data)
            print("✅ Success!")
            return df
        else:
            print("Insufficient data for correlation.")

    except Exception as e:
        print(f"❌ Error: {e}")
        return None

# Test case 3: Different lengths with fix (Success)
print("\n--- Test Case 3: Different lengths with fix ---")
update_correlations_fixed(["SYM1", "SYM2"], mock_data_2)
