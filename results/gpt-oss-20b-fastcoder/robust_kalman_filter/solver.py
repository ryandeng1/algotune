import numpy as np
import cvxpy as cp

class Solver:
    """
    Fast implementation of robust Kalman filtering with Huber loss.
    """

    def solve(self, problem: dict) -> dict:
        # Matrix & vector extraction (numpy)
        A = np.asarray(problem["A"], dtype=float)
        B = np.asarray(problem["B"], dtype=float)
        C = np.asarray(problem["C"], dtype=float)
        y = np.asarray(problem["y"], dtype=float)
        x0 = np.asarray(problem["x_initial"], dtype=float)
        tau = float(problem["tau"])
        M = float(problem["M"])

        N, m = y.shape
        n = A.shape[0]           # state dimension
        p = B.shape[1]           # process noise dimension

        # Decision variables
        x = cp.Variable((N + 1, n))
        w = cp.Variable((N, p))
        v = cp.Variable((N, m))

        # Objective: ‖w‖²₂ + τ Σ Huber(‖v_t‖₂, M)
        obj = (
            cp.sum_squares(w) +  # process noise
            tau * cp.sum(
                cp.huber(cp.norm(v, axis=1), M)
            )
        )

        # Constraints
        constraints = [x[0] == x0]
        constraints.append(x[1:] == A @ x[:-1] + B @ w)          # state update for all t
        constraints.append(y == C @ x[:-1] + v)                  # measurement equations

        # Solve
        prob = cp.Problem(cp.Minimize(obj), constraints)
        prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)

        # Check result and return
        if prob.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} \
                and x.value is not None:
            return {
                "x_hat": x.value.tolist(),
                "w_hat": w.value.tolist(),
                "v_hat": v.value.tolist(),
            }
        else:
            return {"x_hat": [], "w_hat": [], "v_hat": []}