import numpy as np
from scipy.special import xlogy
import cvxpy as cp

class Solver:
    def solve(self, problem: dict) -> dict:
        """Solve the CVXPY optimization in a compact, fast way."""
        P = np.array(problem['P'])
        n, m = P.shape

        # Decide variable dimension
        x = cp.Variable(n, name="x")
        y = P @ x

        # Entropy terms
        c = np.sum(xlogy(P, P), axis=0) / np.log(2.0)
        mutual_information = c @ x + cp.sum(cp.entr(y) / np.log(2.0))

        # Constraints
        constraints = [cp.sum(x) == 1, x >= 0]
        prob = cp.Problem(cp.Maximize(mutual_information), constraints)

        try:
            prob.solve(solver=cp.SCS, verbose=False, eps=1e-8)
        except Exception:  # pragma: no cover
            return None

        if prob.value is None:
            return None
        return {"x": x.value.tolist(), "C": float(prob.value)}