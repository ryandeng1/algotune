# solver.py
import math
from typing import Dict
import cvxpy as cp
import numpy as np
from scipy.special import xlogy

class Solver:
    """
    Maximize mutual information for a discrete memoryless channel.
    The problem is:     max_x   sum_i ( sum_j P_{j,i} log2(P_{j,i} / (P x)_j) )
                      s.t.   sum_i x_i = 1,  x_i >= 0
    where P is an (m × n) probability transition matrix.
    """

    def solve(self, problem: Dict[str, np.ndarray]) -> Dict[str, object] | None:
        P = np.array(problem["P"], dtype=np.float64)
        # shape must be (m, n) with m rows (outputs) and n columns (inputs)
        m, n = P.shape
        if not (m > 0 and n > 0):
            return None

        # pre‑compute constants
        log2 = math.log(2.0)
        # log2(P) with zeros replaced by 0 (log2(0) is -∞ but xlogy handles it)
        log2P = np.zeros_like(P)
        nonzero = P > 0
        log2P[nonzero] = np.log2(P[nonzero])

        # Numpy vector for column‑wise sums of P * log2(P)
        # c_j = sum_i P_{i,j} log2(P_{i,j})
        c = np.sum(P * log2P, axis=0)

        # CVXPY variables
        x = cp.Variable(n)
        y = P @ x

        # Mutual information expressed in bits
        mutual_info = c @ x + cp.sum(cp.entr(y) / log2)

        prob = cp.Problem(cp.Maximize(mutual_info), [cp.sum(x) == 1, x >= 0])

        # Fast solver: SCS (default) with fewer iterations
        try:
            prob.solve(solver=cp.SCS, verbose=False, max_iters=2000)
        except cp.SolverError:
            return None
        except Exception:
            return None

        if prob.value is None:
            return None

        return {"x": x.value.tolist(), "C": prob.value}