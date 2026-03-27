from typing import Any, Dict, List
import numpy as np


class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[float]]:
        """
        Solve the LP centering problem

            minimize   cᵀx – ∑ log(xᵢ)
            subject to Ax = b,  x > 0

        using a Newton method for the Lagrange multipliers.
        """
        c = np.asarray(problem["c"], dtype=np.float64)
        A = np.asarray(problem["A"], dtype=np.float64)
        b = np.asarray(problem["b"], dtype=np.float64)

        n = c.size
        m = b.size

        # initial guess for the dual variables
        lam = np.zeros(m, dtype=np.float64)

        # Newton iterations
        for _ in range(25):
            # compute primal variables from current dual estimate
            aux = c + A.T @ lam
            if np.any(aux <= 0):
                # numerical issue -> slightly perturb
                aux = np.where(aux <= 0, 1e-8, aux)

            x = 1.0 / aux
            # residual of the equality constraints
            g = A @ x - b

            # stop if converged
            if np.linalg.norm(g, ord=np.inf) < 1e-12:
                break

            # Jacobian: -A * diag(x^2) * Aᵀ
            # We avoid explicit dense matrices by solving J dlam = -g
            # J = -(A @ (x**2 * (A.T @ z))) for any z
            # However we can construct the m×m matrix directly as it's small
            X2 = x ** 2
            J = -(A @ (X2[:, None] * A.T))  # shape (m, m)

            # solve for Newton step
            dlam = np.linalg.solve(J, -g)
            lam += dlam

        # final primal solution
        x = 1.0 / (c + A.T @ lam)
        return {"solution": x.tolist()}