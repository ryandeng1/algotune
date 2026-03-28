from typing import Any
import ot
import numpy as np

class Solver:
    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, Any]:
        """
        Solve the EMD problem using ot.lp.emd as quickly as possible.
        """
        a = problem['source_weights']
        b = problem['target_weights']
        M = problem['cost_matrix']

        # ot.lp.emd accepts any ndarray; forcing a contiguous float array is not needed
        # and avoids an extra copy.
        G = ot.lp.emd(a, b, M, check_marginals=False)

        return {'transport_plan': G}