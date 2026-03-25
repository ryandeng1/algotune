import numpy as np
from numba import njit

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        A = np.array(problem["A"])
        B = np.array(problem["B"])
        Q = np.array(problem["Q"])
        R = np.array(problem["R"])
        P = np.array(problem["P"])
        T = problem["T"]
        x0 = np.array(problem["x0"])
        
        n, m = B.shape
        
        S = np.zeros((T + 1, n, n))
        S[T] = P
        
        @njit
        def backward_step(t, A, B, Q, R, S_next, S):
            M1 = R + B.T @ S_next @ B
            M2 = B.T @ S_next @ A
            K = np.linalg.solve(M1, M2, assume_a='pos')
            Acl = A - B @ K
            S_current = Q + K.T @ R @ K + Acl.T @ S_next @ Acl
            S_current = (S_current + S_current.T) / 2
            S[t] = S_current
            return K
        
        K_arr = np.zeros((T, m, n))
        for t in range(T - 1, -1, -1):
            backward_step(t, A, B, Q, R, S[t + 1], S)
        
        U = np.zeros((T, m))
        x = x0
        for t in range(T):
            u = -K_arr[t] @ x
            U[t] = u
            x = A @ x + B @ u
        
        return {"U": U}
