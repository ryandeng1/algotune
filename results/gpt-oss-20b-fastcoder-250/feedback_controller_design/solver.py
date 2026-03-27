import numpy as np
import cvxpy as cp

class Solver:
    def solve(self, problem: dict[str, any]) -> dict[str, any]:
        A = np.asarray(problem["A"], dtype=float)
        B = np.asarray(problem["B"], dtype=float)
        n, m = A.shape[0], B.shape[1]

        # Create variables once
        Q = cp.Variable((n, n), symmetric=True)
        L = cp.Variable((m, n))

        eye_n = np.eye(n)
        eye_2n = np.eye(2 * n)

        # Build constraints
        M1 = cp.bmat([[Q, Q @ A.T + L.T @ B.T],
                      [A @ Q + B @ L, Q]])
        constraints = [M1 >> eye_2n, Q >> eye_n]

        # Solve SDP
        prob = cp.Problem(cp.Minimize(0), constraints)
        prob.solve(solver=cp.CLARABEL)

        # Interpret result
        if prob.status in {"optimal", "optimal_inaccurate"}:
            Q_val = Q.value
            L_val = L.value
            P = np.linalg.inv(Q_val)
            K = L_val @ P
            return {"is_stabilizable": True, "K": K.tolist(), "P": P.tolist()}
        else:
            return {"is_stabilizable": False, "K": None, "P": None}