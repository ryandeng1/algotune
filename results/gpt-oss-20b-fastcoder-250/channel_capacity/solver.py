import math
from typing import Any, Dict, List
import numpy as np
from scipy.optimize import minimize


class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimise the input distribution of a discrete memoryless channel
        to maximise the mutual information.
        """
        # Extract channel transition matrix
        P = np.array(problem.get("P"), dtype=float)
        if P.ndim != 2:
            return None

        m, n = P.shape
        # The solver in the reference expects a square matrix (P.shape == (n, m))
        # which implies a square channel (same #inputs and outputs).
        # If that is not the case we simply refuse to solve.
        if m != n:
            return None
        if m <= 0 or n <= 0:
            return None
        # All columns of P should sum to 1 (row-normalised), but the
        # original code checked columns.  We follow that behaviour.
        if not np.allclose(np.sum(P, axis=0), 1, atol=1e-8):
            return None

        # Pre‑compute constants for the objective
        #  c[i] = (sum_{j} P[j,i] * log2(P[j,i]) )  (in base 2)
        with np.errstate(divide="ignore", invalid="ignore"):
            logP = np.where(P > 0, np.log2(P), 0.0)
        c = np.sum(P * logP, axis=0)  # shape (n,)

        def objective(p: np.ndarray) -> float:
            """
            Negative mutual information (to be minimised):
            I(X;Y) = sum_i p[i] * c[i] - sum_y y * log2(y)
            where y = P @ p  (vector of output probabilities)
            """
            y = P @ p
            # entropy term, use only positive y entries
            idx = y > 0
            h = -np.sum(y[idx] * np.log2(y[idx]))
            return -(np.dot(c, p) + h)

        # Constraints: probabilities sum to 1, each element >= 0
        cons = (
            {"type": "eq", "fun": lambda p: np.sum(p) - 1},
            {"type": "ineq", "fun": lambda p: p},  # ensures p >= 0
        )

        # Initial guess: uniform distribution
        x0 = np.full(n, 1.0 / n)

        try:
            res = minimize(
                objective,
                x0,
                method="SLSQP",
                constraints=cons,
                bounds=[(0.0, 1.0) for _ in range(n)],
                options={"ftol": 1e-9, "maxiter": 1000},
            )
        except Exception:
            return None

        if not res.success:
            return None

        # Numerical errors may lead to tiny negative values; clip to 0
        x_opt = np.clip(res.x, 0, 1)
        # Renormalise to avoid accumulation errors
        x_opt /= x_opt.sum()

        val = -objective(x_opt)  # actual mutual information

        return {"x": x_opt.tolist(), "C": val}