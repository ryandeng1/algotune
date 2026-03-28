import numpy as np
import ot

class Solver:
    def solve(self, problem: dict[str, any]) -> dict[str, any]:
        # Prepare inputs (fast conversion)
        a = np.asarray(problem["source_weights"], dtype=np.float64)
        b = np.asarray(problem["target_weights"], dtype=np.float64)
        M = np.asarray(problem["cost_matrix"], dtype=np.float64, order="C")
        reg = float(problem["reg"])
        try:
            G = ot.sinkhorn(a, b, M, reg)
            if not np.isfinite(G).all():
                raise ValueError("Non‑finite values in transport plan")
        except Exception as exc:
            return {"transport_plan": None, "error_message": str(exc)}
        return {"transport_plan": G, "error_message": None}