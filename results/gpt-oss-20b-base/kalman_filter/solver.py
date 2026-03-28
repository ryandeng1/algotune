import numpy as np

class Solver:
    def solve(self, problem: dict) -> dict:
        A = np.array(problem['A'])
        B = np.array(problem['B'])
        C = np.array(problem['C'])
        y = np.array(problem['y'])
        x0 = np.array(problem['x_initial'])
        tau = float(problem['tau'])

        N, m = y.shape
        n = A.shape[1]
        p = B.shape[1]

        # Build mapping from w to all states x
        # X = T x0 + Gw * w_vec
        T = [np.eye(n)]
        for _ in range(N):
            T.append(A @ T[-1])
        T = np.stack(T, axis=0)          # (N+1, n, n)
        T0 = T @ x0                     # (N+1, n)

        # Gw: (N+1, n, N*p)
        Gw = np.zeros((N + 1, n, N * p))
        for t in range(N):
            Gw[t + 1, :, t * p : (t + 1) * p] = B
            for k in range(t):
                Gw[t + 1, :, k * p : (k + 1) * p] = A @ Gw[t, :, k * p : (k + 1) * p]

        Gw = Gw.reshape((N + 1) * n, N * p)

        # Build measurement mapping: y_vec = (C @ X) + v
        # Here v = y - C @ X
        Cmat = np.vstack([C @ np.eye(n) for _ in range(N)])
        Cx0 = Cmat @ T0.reshape(-1)
        Gvv = Cmat @ Gw

        y_vec = y.reshape(-1)

        # Solve (I + tau Gvv^T Gvv) w = tau Gvv^T (y - Cx0)
        M = np.eye(N * p) + tau * Gvv.T @ Gvv
        rhs = tau * Gvv.T @ (y_vec - Cx0)
        try:
            w_vec = np.linalg.solve(M, rhs)
        except np.linalg.LinAlgError:
            return {'x_hat': [], 'w_hat': [], 'v_hat': []}

        # Recover states
        X_vec = T0.reshape(-1) + Gw @ w_vec
        X = X_vec.reshape(N + 1, n)

        # Recover w
        w = w_vec.reshape(N, p)

        # Recover v
        v = y - (C @ X[:N].T).T

        return {
            'x_hat': X.tolist(),
            'w_hat': w.tolist(),
            'v_hat': v.tolist()
        }