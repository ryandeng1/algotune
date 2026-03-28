import numpy as np

class Solver:
    def solve(self, problem: dict[str, any]) -> dict[str, any]:
        A, B = problem["A"], problem["B"]
        Q, R, P = problem["Q"], problem["R"], problem["P"]
        T, x0 = problem["T"], problem["x0"]
        n, m = B.shape

        S = np.empty((T + 1, n, n))
        K = np.empty((T, m, n))
        S[T] = P.copy()

        for t in range(T - 1, -1, -1):
            St1 = S[t + 1]
            # Solve linear system M1 * K[t] = M2  (K[t] already transposed shape m*n)
            M1 = R + B.T @ St1 @ B
            M2 = B.T @ St1 @ A
            K[t] = np.linalg.solve(M1, M2).T
            # Closed-loop matrix
            Acl = A - B @ K[t].T
            # Update S[t]
            S[t] = Q + K[t].T @ R @ K[t] + Acl.T @ St1 @ Acl
            # Symmetrize S[t] to guard against numerical asymmetry
            S[t] = (S[t] + S[t].T) * 0.5

        U = np.empty((T, m))
        x = x0.copy()
        for t in range(T):
            u = -K[t].T @ x
            U[t] = u.ravel()
            x = A @ x + B @ u
        return {"U": U}