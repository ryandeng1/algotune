import math
import numpy as np
from scipy.special import xlogy
import cvxpy as cp

class Solver:

    def solve(self, problem: dict) -> dict:
        """
        Optimises a convex mutual‐information objective

                max_x   ∑_i c_i x_i  +  ∑_j  entropy(∑_i P[j,i] x_i)

        subject to  x ≥ 0,  1ᵀx = 1.

        The constants c_i are the entropies of the columns of P, fixed in
        advance.  The problem is convex (sum of an affine function and
        the convex function  y → Σ_j entropy(y_j)).  CVXPy with the
        ECOS solver (which works on the dual of the problem) gives very
        fast solutions for the sizes that typically occur in the
        challenges.

        The code below is intentionally minimal and avoids the
        overhead of unnecessary type checking.  It prepares the data,
        builds the CVXPy problem once, and solves it with ECOS set to
        very low verbosity.  The solver is tol=1e-9 and the maximum
        iterations are 10 000 so that the solution is accurate but
        still fast.  If the solver fails for any reason we simply
        return ``None``.
        """
        P = np.asarray(problem['P'], dtype=np.float64)
        if P.ndim != 2:
            return None
        n, m = P.shape
        if n == 0 or m == 0:
            return None

        # Pre‑compute constants
        # c_i = Σ_j P[j,i] log P[j,i]   (base 2)
        #   = Σ_j xlogy(P[j,i], P[j,i]) / ln 2
        #  The second sum in the objective is independent of x except
        #  through y = P x, but the entropic part is convex in y and
        #  therefore in x.
        const_cols = np.sum(xlogy(P, P), axis=0) / math.log(2.0)   # shape (m,)

        x = cp.Variable(m)
        y = P @ x                       # shape (n,)
        mutual = const_cols @ x + cp.sum(cp.entr(y) / math.log(2.0))
        objective = cp.Maximize(mutual)
        constraints = [cp.sum(x) == 1, x >= 0]
        prob = cp.Problem(objective, constraints)

        # Solve with ECOS (dual form) – highly efficient for this type.
        try:
            prob.solve(solver=cp.ECOS,
                       verbose=False,
                       use_indirect=True,   # uses the dual problem internally
                       abstol=1e-9,
                       reltol=1e-9,
                       feastol=1e-9,
                       max_iters=10000)
        except Exception:
            return None

        if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            return None

        return {'x': x.value.ravel().tolist(), 'C': prob.value}