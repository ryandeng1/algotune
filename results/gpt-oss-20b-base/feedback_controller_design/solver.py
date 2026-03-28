import numpy as np
import cvxpy as cp

class Solver:
    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, np.ndarray | None]:
        """Design a state‑feedback controller using an SDP.

        The problem solved is:
            minimize 0
            subject to
              [ Q          Q Aᵀ + Lᵀ Bᵀ ]
              [ A Q + B L   Q        ]  ⪰ I_{2n}
              Q ⪰ I_n
        where Q  is a symmetric n×n matrix and L is an m×n matrix.
        If the problem is feasible, the feedback gain K = L Q⁻¹ is returned
        together with the Lyapunov matrix P = Q⁻¹ and a status flag.
        """
        A = np.asarray(problem["A"])
        B = np.asarray(problem["B"])
        n, m = A.shape[0], B.shape[1]

        Q = cp.Variable((n, n), symmetric=True)
        L = cp.Variable((m, n))

        # Block matrix inequality
        block = cp.bmat([[Q, Q @ A.T + L.T @ B.T],
                         [A @ Q + B @ L, Q]])
        constraints = [block >> np.eye(2 * n), Q >> np.eye(n)]

        prob = cp.Problem(cp.Minimize(0), constraints)
        prob.solve(solver=cp.CLARABEL, warm_start=True)

        if prob.status in {"optimal", "optimal_inaccurate"}:
            Q_val = np.array(Q.value)
            L_val = np.array(L.value)
            K = L_val @ np.linalg.inv(Q_val)
            P = np.linalg.inv(Q_val)
            return {"is_stabilizable": True, "K": K.tolist(), "P": P.tolist()}
        else:
            return {"is_stabilizable": False, "K": None, "P": None}