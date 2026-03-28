import numpy as np
from scipy.linalg import solve as linalg_solve

class Solver:
    def solve(self, problem: dict[str, list[np.ndarray | int]]) -> dict[str, np.ndarray]:
        """
        Compute optimal control sequence via backward Riccati recursion.
        Returns dict with key "U" (shape (T, m)).
        """
        A, B = problem['A'], problem['B']
        Q, R, P = problem['Q'], problem['R'], problem['P']
        T, x0 = problem['T'], problem['x0']

        n, m = B.shape
        #  Riccati matrix and feedback gain
        S = np.empty((T + 1, n, n))
        K = np.empty((T, m, n))
        S[T] = P

        # Backward pass
        for t in range(T - 1, -1, -1):
            St1 = S[t + 1]
            # compute Q_{t+1} = R + Bᵀ S_{t+1} B and S_t
            M1 = R + B.T @ St1 @ B
            M2 = B.T @ St1 @ A
            try:
                K[t] = linalg_solve(M1, M2, assume_a='pos')
            except np.linalg.LinAlgError:            # fallback to pseudo‑inverse
                K[t] = np.linalg.pinv(M1) @ M2
            # closed‑loop dynamics
            Acl = A - B @ K[t]
            # update Riccati matrix
            St = Q + K[t].T @ R @ K[t] + Acl.T @ St1 @ Acl
            S[t] = (St + St.T) * 0.5

        # Forward pass (simulation)
        U = np.empty((T, m))
        x = x0
        for t in range(T):
            u = -K[t] @ x
            U[t] = u.reshape(-1)
            x = A @ x + B @ u

        return {'U': U}