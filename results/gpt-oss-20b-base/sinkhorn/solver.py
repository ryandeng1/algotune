# Optimised implementation – minimal overhead, no unnecessary checks

from typing import Any
import numpy as np
import ot

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        # Convert inputs to contiguous float64 arrays – sinkhorn expects this
        a = np.asarray(problem['source_weights'], dtype=np.float64)
        b = np.asarray(problem['target_weights'], dtype=np.float64)
        M = np.ascontiguousarray(problem['cost_matrix'], dtype=np.float64)
        reg = float(problem['reg'])
        try:
            G = ot.sinkhorn(a, b, M, reg)
            return {'transport_plan': G, 'error_message': None}
        except Exception as exc:
            return {'transport_plan': None, 'error_message': str(exc)}