import math
from typing import Any, Dict
import numpy as np
import cvxpy as cp
from scipy.special import xlogy

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any] | None:
        # Extract channel matrix P and ensure shape is (n, m)
        P = np.asarray(problem.get("P", []), dtype=float)
        if P.ndim != 2:
            return None
        m, n = P.shape
        if m != n:
            return None
        if n == 0:
            return None

        # Variable for input distribution
        x = cp.Variable(shape=n, name="x")

        # Output symbol probabilities
        y = P @ x

        # Precompute constants: c_i = ∑_j P_ij log_2(P_ij)
        # Using xlogy to avoid log(0)
        c = np.sum(xlogy(P, P), axis=0) / math.log(2)

        # Mutual information: ∑_i c_i x_i + ∑_j entr(y_j)/log(2)
        mutual_information = c @ x + cp.sum(cp.entr(y)) / math.log(2)

        # Objective and constraints
        constraints = [cp.sum(x) == 1, x >= 0]
        objective = cp.Maximize(mutual_information)

        # Solve the convex program
        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=cp.SCS, verbose=False)
        except (cp.SolverError, Exception):
            return None

        # Result handling
        if prob.value is None:
            return None

        return {"x": x.value.tolist(), "C": float(prob.value)}