import numpy as np


class Solver:
    def solve(self, problem: dict) -> dict:
        # extract data
        A = np.asarray(problem["A"], dtype=float)
        B = np.asarray(problem["B"], dtype=float)
        C = np.asarray(problem["C"], dtype=float)
        y = np.asarray(problem["y"], dtype=float)
        x0 = np.asarray(problem["x_initial"], dtype=float)
        tau = float(problem["tau"])

        N, m = y.shape
        n = A.shape[1]
        p = B.shape[1]

        # Pre‑compute powers of A and intermediate matrices
        A_pow = [np.eye(n, dtype=float)]
        for _ in range(1, N + 1):
            A_pow.append(A @ A_pow[-1])

        # Build B_tilde such that
        # x[t] = A_pow[t] @ x0 + B_tilde[t] @ w
        B_tilde = np.zeros((N, n, N, p), dtype=float)  # not used directly
        Psi = np.zeros((n * N, p * N), dtype=float)   # slope of [x0,x1,...,x_{N-1}] wrt w

        for t in range(N):
            # coefficient matrix for w[k] in x[t]
            for k in range(t):
                idx_w = k * p
                idx_x = t * n
                Psi[idx_x : idx_x + n, idx_w : idx_w + p] = (
                    A_pow[t - k - 1] @ B
                )

        # Vector part due to x0
        Phi_x0 = np.concatenate([A_pow[t] @ x0 for t in range(N)], axis=0)  # shape (n*N,)

        # Build the linear operator on w: (y - C * Phi_x0) - C * Psi * w
        Q = C @ Psi                  # shape (m*N, p*N)
        r = y - C @ Phi_x0           # shape (m*N,)

        # Solve weighted least squares: w = argmin ||w||^2 + tau * ||Q w - r||^2
        # Equivalent to least squares on [I; sqrt(tau)*Q] * w ~ [0; sqrt(tau)*r]
        I = np.eye(p * N)
        sqrt_tau = np.sqrt(tau)
        A_ls = np.vstack([I, sqrt_tau * Q])
        b_ls = np.concatenate([np.zeros(p * N), sqrt_tau * r])

        # Use numpy.linalg.lstsq (economical) for stability
        try:
            w_hat, *_ = np.linalg.lstsq(A_ls, b_ls, rcond=None)
        except Exception:
            return {"x_hat": [], "w_hat": [], "v_hat": []}

        # recover x sequentially
        x_hat = np.zeros((N, n))
        prev = x0
        for t in range(N):
            w_t = w_hat[t * p : (t + 1) * p]
            x_t = A @ prev + B @ w_t
            x_hat[t] = x_t
            prev = x_t

        # recover v from measurements: v[t] = y[t] - C x[t]
        v_hat = y - (C @ x_hat.T).T

        return {
            "x_hat": x_hat.tolist(),
            "w_hat": w_hat.reshape((N, p)).tolist(),
            "v_hat": v_hat.tolist(),
        }