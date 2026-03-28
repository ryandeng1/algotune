from typing import Any
import numpy as np
import ot

class Solver:
    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, list[list[float]]]:
        a = problem["source_weights"]
        b = problem["target_weights"]
        M = problem["cost_matrix"]

        M_cont = np.ascontiguousarray(M, dtype=np.float64)
        G = ot.lp.emd(a, b, M_cont, check_marginals=False)
        return {"transport_plan": G.tolist()}