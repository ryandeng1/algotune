import numpy as np

class Solver:
    def solve(self, problem: dict[str, list]) -> dict[str, list]:
        """
        Solve the lp centering problem:
            minimize  cᵀx - Σ log(x)
            subject to  A x = b ,  x > 0
        using the KKT conditions and a Newton method for the
        dual variables.  The method is purely NumPy based and
        therefore significantly faster than a generic CVXPY call.
        """
        c = np.asarray(problem["c"], dtype=np.float64)
        A = np.asarray(problem["A"], dtype=np.float64)
        b = np.asarray(problem["b"], dtype=np.float64)

        m, n = A.shape
        # initial dual variable (m-dim)
        lam = np.zeros(m, dtype=np.float64)

        # Newton iterations for λ
        max_iter = 100
        tol = 1e-10
        for _ in range(max_iter):
            # Compute denominators d_i = c_i - (Aᵀ λ)_i
            A_T_lam = A.T @ lam
            d = c - A_T_lam
            # if any denominator <= 0, add small shift and break
            if np.any(d <= 0):
                d = np.maximum(d, 1e-12)

            # Current primal variables
            x = 1.0 / d

            # Residual g(λ) = A x - b
            g = A @ x - b
            if np.linalg.norm(g, ord=np.inf) < tol:
                break

            # Jacobian J = -A * diag(d^{-2}) * Aᵀ
            w = d ** -2
            # Compute J * v efficiently: Jv = -A * (w * (Aᵀ v))
            def Jv(v):
                return -A @ (w * (A.T @ v))

            # Solve linear system J * delta = g
            # Using a simple linear solver via np.linalg.solve on J explicitly
            # J is symmetric negative definite; we can use np.linalg.solve on dense matrix
            J = -A @ np.diag(w) @ A.T
            delta = np.linalg.solve(J, g)

            lam -= delta

        # Final primal solution
        A_T_lam = A.T @ lam
        d = c - A_T_lam
        x = 1.0 / d
        return {"solution": x.tolist()}