from typing import Any, Dict
import numpy as np
from scipy.linalg import solve_continuous_lyapunov, eigvals

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Checks asymptotic stability of the linear system ˙x = A x
        and, if stable, returns a corresponding Lyapunov matrix P
        that satisfies AᵀP + PA = -I and P ≻ 0.

        Parameters
        ----------
        problem : dict
            Dictionary containing the system matrix with key 'A'.

        Returns
        -------
        dict
            {'is_stable': bool, 'P': list or None}
        """
        A = np.asarray(problem['A'], dtype=float)
        n = A.shape[0]

        # 1. Quick spectral test: all eigenvalues must have negative real part.
        eigs = eigvals(A)
        if np.any(eigs.real >= 0):
            return {'is_stable': False, 'P': None}

        # 2. Solve continuous Lyapunov equation AᵀP + PA = -I
        try:
            P = solve_continuous_lyapunov(A.T, -np.eye(n))
        except Exception:
            return {'is_stable': False, 'P': None}

        # 3. Verify positive definiteness (optional, numeric stability)
        if np.all(np.linalg.eigvals(P) > 1e-8):
            return {'is_stable': True, 'P': P.tolist()}
        else:
            # Numerical noise: attempt a symmetric part
            P_sym = (P + P.T) / 2
            if np.all(np.linalg.eigvals(P_sym) > 1e-8):
                return {'is_stable': True, 'P': P_sym.tolist()}
            return {'is_stable': False, 'P': None}