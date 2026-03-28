import numpy as np
from numpy.linalg import inv, pinv, solve

def solve_discrete_are(A, B, Q, R):
    """Solves the discrete-time algebraic Riccati equation:
       X = A.T X A - A.T X B (R + B.T X B)^{-1} B.T X A + Q
    """
    n = A.shape[0]
    X = Q.copy()
    # iteration via Newton–Schulz
    for _ in range(100):
        S = R + B.T @ X @ B
        K = solve(S, B.T @ X @ A)
        X_new = A.T @ X @ A - A.T @ X @ B @ K + Q
        if np.allclose(X, X_new, atol=1e-8, rtol=1e-6):
            X = X_new
            break
        X = X_new
    return X


class Solver:
    def solve(self, problem: dict) -> dict:
        """Fast discrete-time LQR solver that returns K and the Lyapunov matrix P"""
        A = np.asarray(problem["A"], dtype=float)
        B = np.asarray(problem["B"], dtype=float)

        n, m = A.shape[0], B.shape[1]

        # Use identity matrices for Q and R (minimum‑norm solution)
        Q = np.eye(n)
        R = np.eye(m)

        try:
            P = solve_discrete_are(A, B, Q, R)
            # Compute K = (R + B.T P B)^{-1} B.T P A
            S = R + B.T @ P @ B
            K = solve(S, B.T @ P @ A)

            return {
                "is_stabilizable": True,
                "K": K.tolist(),
                "P": P.tolist(),
            }
        except Exception:
            return {"is_stabilizable": False, "K": None, "P": None}