from typing import Any, Dict, List
import cvxpy as cp
import numpy as np

# The solver deliberately uses only the most performant features of cvxpy
# and turns off unnecessary diagnostics to cut down on overhead.

class Solver:
    """
    Robust Kalman filter using the Huber loss on the measurement noise.
    The implementation is carefully tuned for speed:
    - vectorised Problem construction
    - minimal Python/CVXPY overhead
    - deterministic solver choice
    """

    def __init__(self) -> None:
        # Pick a fast solver that supports the used features.
        # OSQP handles least‑squares objective efficiently.
        # Avoid solver warnings that slow down the solve step.
        self._solver_opts = {"silent": True}

    def solve(self, problem: Dict[str, Any]) -> Dict[str, List]:
        """
        Solve the robust Kalman filtering problem.

        Parameters
        ----------
        problem : dict
            Should contain the keys:
                - 'A'  : (n,n) state transition matrix
                - 'B'  : (n,p) control matrix
                - 'C'  : (m,n) measurement matrix
                - 'y'  : (N,m) measurement sequence
                - 'x_initial' : initial state (n,)
                - 'tau' : scalar weighting for Huber term
                - 'M'   : Huber threshold

        Returns
        -------
        dict with keys 'x_hat', 'w_hat', 'v_hat', each being a nested list
        representation of the corresponding numpy arrays.
        """
        # ----- data extraction -------------------------------------------------
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

        # ----- CVXPY variables -----------------------------------------------
        x = cp.Variable((N + 1, n), name="x")
        w = cp.Variable((N, p), name="w")
        v = cp.Variable((N, m), name="v")

        # ----- objective terms -----------------------------------------------
        # Least‑squares on process noise
        process_noise = cp.sum_squares(w)

        # Huber on measurement noise, vectorised over the N steps
        #  cp.norm(v[i,:]) is not vectorisable, so we sum over the axis.
        huber_terms = cp.huber(cp.norm(v, axis=1), M)
        measurement_noise = tau * cp.sum(huber_terms)

        obj = cp.Minimize(process_noise + measurement_noise)

        # ----- constraints -----------------------------------------------
        constraints = [x[0] == x0]
        constraints.append(x[1:] == A @ x[:-1] + B @ w)
        constraints.append(y == C @ x[:-1] + v)

        # ----- problem definition -------------------------------------------
        prob = cp.Problem(obj, constraints)

        # ----- solve ---------------------------------------------------------
        try:
            prob.solve(solver=cp.OSQP, **self._solver_opts)
        except Exception:
            # Any error leads to an empty result
            return {"x_hat": [], "w_hat": [], "v_hat": []}

        # If the solver did not succeed, also return empty lists
        if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or x.value is None:
            return {"x_hat": [], "w_hat": [], "v_hat": []}

        # ----- result conversion ---------------------------------------------
        return {
            "x_hat": x.value.tolist(),
            "w_hat": w.value.tolist(),
            "v_hat": v.value.tolist(),
        }