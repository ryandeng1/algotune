import math
import numpy as np
import cvxpy as cp
from scipy.special import xlogy
from typing import Dict, Any

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Solve the input–output maximisation problem

          maximise  C = sum_i x_i log_2(Σ_j P[j,i] x_j) - Σ_j p_j log_2 p_j
          subject to  x >= 0 , Σ_i x_i = 1

        where p = Σ_j P[:,j] x_j and P is a m×n transition matrix of a channel.
        The function returns a dictionary ``{"x": [x_0,...,x_{n-1}], "C": C}``
        or ``None`` if the problem is ill‑formed or CVXPY fails.
        """
        try:
            P = np.asarray(problem["P"], dtype=np.float64)
        except (KeyError, ValueError, TypeError):
            return None

        if P.ndim != 2:
            return None

        m, n = P.shape
        if n == 0 or m == 0:
            return None

        # P must be column‑stochastic (columns sum to 1)
        if not np.allclose(P.sum(axis=0), 1.0, atol=1e-12):
            return None

        # constant term independent of x:  Σ_i x_i * Σ_j P[j,i] log_2 P[j,i]
        # The inner sum is independent of the optimisation variable x and
        # therefore can be computed once.
        const_term = np.sum(xlogy(P, P), axis=0) / math.log(2)

        # optimisation variable
        x = cp.Variable(n, name="x")

        # channel output probabilities
        y = P @ x

        # mutual information (negative entropy + cross‑entropy)
        mutual_information = const_term @ x + cp.sum(cp.entr(y)) / math.log(2)

        objective = cp.Maximize(mutual_information)
        constraints = [cp.sum(x) == 1, x >= 0]

        prob = cp.Problem(objective, constraints)
        try:
            prob.solve()
        except (cp.SolverError, Exception):
            return None

        if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or prob.value is None:
            return None

        return {"x": x.value.tolist(), "C": prob.value}