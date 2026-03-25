from typing import Any
import numpy as np
import numba

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        a = np.array(problem["source_weights"], dtype=np.float64)
        b = np.array(problem["target_weights"], dtype=np.float64)
        M = np.ascontiguousarray(problem["cost_matrix"], dtype=np.float64)
        reg = float(problem["reg"])
        
        n, m = M.shape
        
        @numba.njit
        def sinkhorn_iter(a, b, P):
            for _ in range(100):
                row_sums = np.sum(P, axis=1)
                row_sums = np.maximum(row_sums, 1e-10)
                P = (P * a[:, np.newaxis]) / row_sums[:, np.newaxis]
                
                col_sums = np.sum(P, axis=0)
                col_sums = np.maximum(col_sums, 1e-10)
                P = (P * b[np.newaxis, :]) / col_sums[np.newaxis, :]
            return P
        
        P = np.exp(-M / reg)
        P = sinkhorn_iter(a, b, P)
        
        if not np.isfinite(P).all():
            return {"transport_plan": None, "error_message": "Non-finite values in transport plan"}
        
        return {"transport_plan": P.tolist()}
