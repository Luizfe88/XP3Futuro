import sys, os
sys.path[:] = [p for p in sys.path if "xp3v5" not in (p or "")]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bot import *

if __name__ == "__main__":
    main()
