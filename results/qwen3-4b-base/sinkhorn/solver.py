import numpy as np
import numba

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list[list[float]] | None | str]:
        a = np.array(problem["source_weights"], dtype=np.float64)
        b = np.array(problem["target_weights"], dtype=np.float64)
        M = np.ascontiguousarray(problem["cost_matrix"], dtype=np.float64)
        reg = float(problem["reg"])
        n, m = a.shape[0], b.shape[0]

        @numba.njit
        def sinkhorn(a, b, M, reg, max_iter=1000):
            u = np.zeros(n, dtype=np.float64)
            v = np.zeros(m, dtype=np.float64)
            for _ in range(max_iter):
                P = np.exp(-(M + u[:, np.newaxis] + v[np.newaxis, :]) / reg)
                P_row = np.sum(P, axis=1, keepdims=True)
                P = P / P_row
                P_col = np.sum(P, axis=0, keepdims=True)
                P = P / P_col
                u = u + (a - np.sum(P, axis=1)) / reg
                v = v + (b - np.sum(P, axis=0)) / reg
            return P

        G = sinkhorn(a, b, M, reg, 1000)
        if not np.isfinite(G).all():
            return {"transport_plan": None, "error_message": "Non-finite values in transport plan"}
        return {"transport_plan": G.tolist()}
