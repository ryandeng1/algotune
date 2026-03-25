# solver.py
import numpy as np
import cvxpy as cp


class Solver:
    """Projection onto a CVaR constraint set."""

    def solve(self, problem: dict, **kwargs) -> dict:
        """
        Compute the projection of x0 onto the set {x | CVaR_β(Ax) ≤ κ}.
        The problem is convex and equivalent to
            minimize ||x - x0||₂²
            subject to  sum_largest(Ax, k) ≤ α
        where  k = ceil((1-β) * n_scenarios)
                α = κ * k
        """
        # ---- data extraction ---------------------------------------------
        x0 = np.asarray(problem["x0"], dtype=np.float64)
        A = np.asarray(problem["loss_scenarios"], dtype=np.float64)
        beta = float(problem.get("beta", 0.95))
        kappa = float(problem.get("kappa", 0.0))

        n_scenarios, n_dims = A.shape

        # ---- problem parameters ------------------------------------------
        k = int(np.ceil((1.0 - beta) * n_scenarios))
        # guard against k=0 (β=1) → no constraint
        if k == 0:
            return {"x_proj": x0.tolist()}

        alpha = kappa * k

        # ---- cvxpy formulation -------------------------------------------
        x = cp.Variable(n_dims)
        objective = cp.Minimize(cp.sum_squares(x - x0))
        constraint = cp.sum_largest(A @ x, k) <= alpha
        prob = cp.Problem(objective, [constraint])

        # ---- solve --------------------------------------------------------
        try:
            prob.solve(solver=cp.ECOS, verbose=False, max_iters=1000)
            if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or x.value is None:
                return {"x_proj": []}
            return {"x_proj": x.value.tolist()}
        except Exception:
            return {"x_proj": []}
