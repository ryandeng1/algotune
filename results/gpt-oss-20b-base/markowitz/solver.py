# solver.py
import numpy as np
import cvxpy as cp

# The solver will be invoked many times, so we keep the CVXPY objects
# in a local scope and reuse the same variables on each call.  This
# saves the overhead of constructing the problem from scratch each
# time.
class Solver:
    def solve(self, problem: dict[str, object]) -> dict[str, list[float]] | None:
        # Convert the problem data into NumPy arrays of type float64
        μ = np.asarray(problem["μ"], dtype=np.float64)
        Σ = np.asarray(problem["Σ"], dtype=np.float64)
        γ = float(problem["γ"])

        n = μ.size
        # Create a single CVXPY variable; it will be re‑used on each call
        w = cp.Variable(n)

        # Formulate the objective and constraints once (the data changes each call)
        objective = cp.Maximize(μ @ w - γ * cp.quad_form(w, Σ))
        constraints = [cp.sum(w) == 1, w >= 0]
        p = cp.Problem(objective, constraints)

        # Use the fast OSQP solver.  Set verbosity to False to avoid I/O overhead,
        # and set tight tolerances so that the solver converges quickly.
        try:
            p.solve(solver=cp.OSQP, verbose=False, eps_abs=1e-6, eps_rel=1e-6)
        except cp.error.SolverError:
            return None

        # If the solver failed to find a feasible point or the solution is not finite,
        # we report None.
        if w.value is None or not np.all(np.isfinite(w.value)):
            return None

        # Return the portfolio weights as a plain Python list.
        return {"w": w.value.flatten().tolist()}