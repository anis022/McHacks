import sys
import os
import ctypes
from unittest.mock import MagicMock

# Mock dependencies that might be missing in the dev environment
sys.modules["websocket"] = MagicMock()
sys.modules["requests"] = MagicMock()
sys.modules["urllib3"] = MagicMock()
sys.modules["numpy"] = MagicMock()
# sys.modules["qiskit"] = MagicMock() # Qiskit is handled in the file itself

from student_algorithm import NativeMarketEngine, compile_cpp

def test_integration():
    print("Testing C++ Integration...")
    
    # 1. Test Compilation
    try:
        if os.path.exists("market_engine.so"):
            os.remove("market_engine.so")
        compile_cpp()
        if not os.path.exists("market_engine.so"):
            print("FAIL: market_engine.so not created.")
            return False
        print("PASS: Compilation successful.")
    except Exception as e:
        print(f"FAIL: Compilation raised exception: {e}")
        return False

    # 2. Test Linking and Calculation
    try:
        engine = NativeMarketEngine()
        print("PASS: NativeMarketEngine initialized.")
        
        # Test calculation
        mid = 100.0
        inv = 10
        time_left = 1.0
        
        res = engine.calculate(mid, inv, time_left)
        print(f"Test Result: {res}")
        
        if res['reservation_price'] == 0.0 and res['spread'] == 0.0:
            print("WARNING: Result is all zeros. Check logic.")
        else:
            print("PASS: Calculation returned non-zero values.")
            
    except Exception as e:
        print(f"FAIL: Runtime error: {e}")
        return False

    return True

if __name__ == "__main__":
    if test_integration():
        print("ALL TESTS PASSED")
        sys.exit(0)
    else:
        print("TESTS FAILED")
        sys.exit(1)
