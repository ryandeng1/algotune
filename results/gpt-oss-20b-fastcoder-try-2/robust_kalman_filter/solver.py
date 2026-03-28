from typing import Any

import numpy as np
import cvxpy as cp

# Pre‑allocate the OSQP solver options once – they are reused for every call
_OSQP_OPTS = dict(verbose=False, eps_abs=1e-6, eps_rel=1e-6, max_iter=10000)


class Solver:
    """
    Optimized robust Kalman filter solver using CVXPY with the OSQP solver.
    """

    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Solve the robust Kalman filtering problem with a Huber loss penalty.

        Parameters
        ----------
        problem : dict
            Dictionary containing the system matrices and data. Keys:
                - A (n×n)
                - B (n×p)
                - C (m×n)
                - y (N×m)
                - x_initial (n)
                - tau (scalar)
                - M (scalar)

        Returns
        -------
        dict
            Estimated states (`x_hat`), process noise (`w_hat`), and measurement noise
            (`v_hat`). If the problem is infeasible or an error occurs, empty lists are
            returned.
        """
        # ---------------------------------------------------------------------
        # 1. Parse inputs once – avoid repeated conversions
        # ---------------------------------------------------------------------
        A = np.asarray(problem["A"], dtype=np.float64)
        B = np.asarray(problem["B"], dtype=np.float64)
        C = np.asarray(problem["C"], dtype=np.float64)
        y = np.asarray(problem["y"], dtype=np.float64)
        x0 = np.asarray(problem["x_initial"], dtype=np.float64)

        tau = float(problem["tau"])
        M = float(problem["M"])

        N, m = y.shape
        n = A.shape[0]
        p = B.shape[1]

        # ---------------------------------------------------------------------
        # 2. CVXPY variables
        # ---------------------------------------------------------------------
        x = cp.Variable((N + 1, n), name="x")
        w = cp.Variable((N, p), name="w")
        v = cp.Variable((N, m), name="v")

        # ---------------------------------------------------------------------
        # 3. Objective
        # ---------------------------------------------------------------------
        process_term = cp.sum_squares(w)                    # Σ ||w_t||^2
        meas_term = tau * cp.sum(cp.huber(cp.norm(v, axis=1), M))  # Σ huber(||v_t||, M)
        obj = cp.Minimize(process_term + meas_term)

        # ---------------------------------------------------------------------
        # 4. Constraints
        # ---------------------------------------------------------------------
        constraints = [x[0] == x0]

        # System dynamics: x[t+1] = A*x[t] + B*w[t]
        constraints += [x[t + 1] == A @ x[t] + B @ w[t] for t in range(N)]

        # Measurement model: y[t] = C*x[t] + v[t]
        constraints += [y[t] == C @ x[t] + v[t] for t in range(N)]

        # ---------------------------------------------------------------------
        # 5. Solve
        # ---------------------------------------------------------------------
        prob = cp.Problem(obj, constraints)
        try:
            prob.solve(solver=cp.OSQP, **_OSQP_OPTS)
        except cp.SolverError:
            return {"x_hat": [], "w_hat": [], "v_hat": []}
        except Exception:
            return {"x_hat": [], "w_hat": [], "v_hat": []}

        # ---------------------------------------------------------------------
        # 6. Return results
        # ---------------------------------------------------------------------
        if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or x.value is None:
            return {"x_hat": [], "w_hat": [], "v_hat": []}

        return {
            "x_hat": x.value.tolist(),
            "w_hat": w.value.tolist(),
            "v_hat": v.value.tolist(),
        }