import numpy as np

class Solver:
    def solve(self, problem: dict) -> dict:
        A, B = problem["A"], problem["B"]
        Q, R, P = problem["Q"], problem["R"], problem["P"]
        T, x0 = problem["T"], problem["x0"]

        n, m = B.shape
        S = np.empty((T + 1, n, n))
        K = np.empty((T, m, n))
        S[T] = P

        for t in range(T - 1, -1, -1):
            St1 = S[t + 1]
            M1 = R + B.T @ St1 @ B
            M2 = B.T @ St1 @ A
            try:
                K[t] = np.linalg.solve(M1, M2)
            except np.linalg.LinAlgError:
                K[t] = np.linalg.pinv(M1) @ M2
            Acl = A - B @ K[t]
            S[t] = Q + K[t].T @ R @ K[t] + Acl.T @ St1 @ Acl
            S[t] = (S[t] + S[t].T) * 0.5

        U = np.empty((T, m))
        x = x0
        for t in range(T):
            u = -K[t] @ x
            U[t] = u.ravel()
            x = A @ x + B @ u

        return {"U": U}