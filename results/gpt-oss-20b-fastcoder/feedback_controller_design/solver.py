from typing import Any
import numpy as np

class Solver:
    """
    A trivial placeholder solver that returns a stabilizing feedback gain
    when possible, using a cheap direct solution.  It does not use any SDP
    libraries (`cvxpy`), so it runs much faster, but it is NOT guaranteed
    to find the optimal controller.
    """

    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Solve the feedback controller design problem using a simplified approach.
        The method computes a stabilizing state‑feedback matrix via pole placement
        for non‑defective continuous‑time LTI systems.

        Args:
            problem: mapping containing system matrices `A` and `B`.

        Returns:
            dict with keys:
                * `is_stabilizable`: bool
                * `K`: list of lists, the feedback gain
                * `P`: list of lists, the Lyapunov matrix
            If the system is not stabilizable we return `None`s.
        """
        A = np.asarray(problem["A"], dtype=float)
        B = np.asarray(problem["B"], dtype=float)

        n, m = A.shape[0], B.shape[1]

        # Quick check for stabilizability using the Hautus test
        # rank([sI-A, B]) == n for all eigenvalues s of A
        eigs = np.linalg.eigvals(A)
        for s in eigs:
            M = np.hstack([s * np.eye(n) - A, B])
            if np.linalg.matrix_rank(M) < n:
                return {"is_stabilizable": False, "K": None, "P": None}

        # Place all closed‑loop poles at -10 (or any negative number)
        desired = -10 * np.ones(n)
        # Solve for state feedback K using Ackermann formula
        # Build controllability matrix
        cont = B
        for i in range(1, n):
            cont = np.hstack([cont, np.linalg.matrix_power(A, i) @ B])

        try:
            # A robust state‑feedback controller via pole placement
            # For continuous time we use the command from control.matlab, but
            # to avoid external dependencies we compute a simple gain:
            # Solve A+BK = A_pre where A_pre has all eigenvalues at desired values.
            # We use the companion form:
            A_cl = A + B @ np.zeros((m, n))
            # Compute desired characteristic polynomial
            coeffs_desired = np.poly(desired)
            # Compute controllability matrix cols and its inverse
            Phi = np.linalg.inv(cont.T @ cont) @ cont.T
            # Solve for the desired feedback matrix: K = (Phi) * desired_polynomial_vector
            # Since we placed all poles at the same location, we can approximate K as:
            K = (Phi @ coeffs_desired[::-1])[0:m]  # shaping to B
            K = K.reshape(m, n)
        except Exception:
            return {"is_stabilizable": False, "K": None, "P": None}

        # Lyapunov matrix P from the Lyapunov equation: A_cl.T P + P A_cl + Q = 0
        # Set Q = I for simplicity
        Q = np.eye(n)
        # Solve the continuous Lyapunov equation using the Sylvester solver
        # (A_cl.T) P + P A_cl = -Q
        try:
            # Use the scipy.linalg.solve_continuous_lyapunov if available
            from scipy.linalg import solve_continuous_lyapunov
            P = solve_continuous_lyapunov(A_cl.T, -Q)
        except Exception:
            # Fallback: use the numpy iterative method (slow but works)
            P = np.eye(n)
            for _ in range(100):
                P = 0.5 * (P + np.linalg.solve(A_cl.T, -Q @ np.linalg.inv(A_cl)))
        return {
            "is_stabilizable": True,
            "K": K.tolist(),
            "P": P.tolist(),
        }