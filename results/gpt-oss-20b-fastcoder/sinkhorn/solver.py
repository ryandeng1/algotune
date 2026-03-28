import numpy as np
import ot

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list[list[float]] | None | str]:
        a = np.asarray(problem['source_weights'], dtype=np.float64)
        b = np.asarray(problem['target_weights'], dtype=np.float64)
        M = np.ascontiguousarray(problem['cost_matrix'], dtype=np.float64)
        reg = float(problem['reg'])
        # Directly compute Sinkhorn matrix; any exception will be propagated
        G = ot.sinkhorn(a, b, M, reg)
        return {"transport_plan": G, "error_message": None}