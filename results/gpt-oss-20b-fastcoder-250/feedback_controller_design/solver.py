import numpy as np
from scipy.linalg import svd, solve_discrete_lyapunov
from scipy.signal import place_poles

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Design a static state‑feedback controller K for a discrete–time LTI system
        x_{k+1} = A x_k + B u_k  by checking stabilizability, placing eigenvalues
        inside the unit disc (via pole placement) and computing the corresponding
        Lyapunov matrix P.

        Parameters
        ----------
        problem : dict
            Dictionary containing the numpy‑compatible matrices `A` (n×n) and `B` (n×m).

        Returns
        -------
        dict
            Dictionary with keys:
                is_stabilizable (bool)
                K (m×n numpy array or None)
                P (n×n numpy array or None)
        """
        # Convert inputs to numpy arrays
        A = np.asarray(problem["A"], dtype=float)
        B = np.asarray(problem["B"], dtype=float)
        n, m = A.shape[0], B.shape[1]

        # ---- 1. Check stabilizability ------------------------------------
        # Compute controllability matrix [B, AB, A^2B, ... , A^{n-1}B]
        ctrb_cols = [B]
        AB = A @ B
        for _ in range(1, n):
            ctrb_cols.append(AB)
            AB = A @ AB
        ctrb_matrix = np.hstack(ctrb_cols)
        # Rank via SVD
        rank = np.linalg.matrix_rank(ctrb_matrix, tol=1e-8)
        is_stabilizable = rank == n

        if not is_stabilizable:
            return {"is_stabilizable": False, "K": None, "P": None}

        # ---- 2. Pole placement -------------------------------------------
        # Desired closed‑loop eigenvalues: place them at 0.5 (inside unit circle)
        desired_poles = np.full(n, 0.5, dtype=complex)

        # Use scipy's place_poles for discrete-time systems
        try:
            res = place_poles(A, B, desired_poles, method='YT' if hasattr(place_poles, 'YT') else None)
            K = res.gain_matrix
        except Exception:
            # Fallback: use linear-quadratic regulator for a simple Q=I, R=I
            from scipy.linalg import solve_discrete_are
            Q = np.eye(n)
            R = np.eye(m)
            P_sol = solve_discrete_are(A, B, Q, R)
            K = np.linalg.solve(P_sol, B @ np.linalg.inv(R))

        # ---- 3. Compute Lyapunov matrix ----------------------------------
        A_cl = A + B @ K
        try:
            P = solve_discrete_lyapunov(A_cl.T, np.eye(n))
        except Exception:
            # Fallback: use continuous-time Lyapunov as approximation
            from scipy.linalg import solve_continuous_lyapunov
            P = solve_continuous_lyapunov(A_cl.T, np.eye(n))

        return {
            "is_stabilizable": True,
            "K": K.tolist(),
            "P": P.tolist()
        }
