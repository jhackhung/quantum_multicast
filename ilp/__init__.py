"""CP-SAT optimal baseline for WQMN (see ilp/README.md)."""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:  # 讓 `import Graph / QSTA / evaluate` 在任何入口都可用
    sys.path.insert(0, _ROOT)
