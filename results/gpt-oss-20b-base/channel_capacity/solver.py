import math
import numpy as np
import cvxpy as cp
from scipy.special import xlogy

class Solver:
    """Maximises mutual information I(X;Y)=cᵀx + ⟨entr(Px),1⟩
    with x ≥ 0, 1ᵀx=1.  Input is a problem dict with key 'P'."""

    def solve(self, problem: dict) -> dict:
        try:
            # Convert to numpy array once – all later ops are vectorised.
            P = np.asarray(problem["P"])
            if P.ndim != 2:
                return None
            n, m = P.shape
            if n == 0 or m == 0:
                return None
            # The objective constant term.
            c = np.sum(xlogy(P, P), axis=0) / math.log(2)

            # CVXPy problem construction.
            x = cp.Variable(n, name="x")
            y = P @ x
            obj = cp.Maximize(c @ x + cp.sum(cp.entr(y) / math.log(2)))
            cons = [cp.sum(x) == 1, x >= 0]
            prob = cp.Problem(obj, cons)

            # Solve with default solver; silence output for speed.
            prob.solve(verbose=False, warm_start=True, max_iters=1000)

            if prob.status not in {"optimal", "optimal_inaccurate"}:
                return None
            return {"x": x.value.tolist(), "C": prob.value}
        except Exception:
            return None