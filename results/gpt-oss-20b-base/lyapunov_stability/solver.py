import numpy as np
from scipy.linalg import solve_discrete_lyapunov

class Solver:

    def solve(self, problem: dict) -> dict:
        """
        Directly solves the discrete Lyapunov equation
        Aᵀ P A − P = −I
        and tests whether P is positive definite.
        """
        A = np.array(problem['A'], dtype=float)
        try:
            # Solve the discrete Lyapunov equation
            P = solve_discrete_lyapunov(A.T, np.eye(A.shape[0]))
        except Exception as exc:
            # Numerical failure → not stable
            return {'is_stable': False, 'P': None}

        # Check positive definiteness: all eigenvalues > eps
        eigs = np.linalg.eigvalsh(P)
        if np.all(eigs > 1e-8):
            return {'is_stable': True, 'P': P.tolist()}
        else:
            return {'is_stable': False, 'P': None}