# Optimised solver for Lyapunov stability analysis
# Uses cvxpy with the fast SCS solver and minimal overhead
# Author: ChatGPT – Performance Engineer

from __future__ import annotations

import numpy as np
import cvxpy as cp
from typing import Any, Dict, List, Tuple, Optional

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Determine asymptotic stability of a discrete-time linear system
        A x(t+1) = A x(t) by solving the Lyapunov inequalities

            P  » I        (positive definite)
            A^T P A  <≪  P - I

        A stable system exists a solution to the above if and only if
        all eigenvalues of A are strictly within the unit circle.

        Parameters
        ----------
        problem : dict
            Dictionary that contains the matrix `A` as a numeric 2‑D list or array.

        Returns
        -------
        dict
            ``{'is_stable': bool, 'P': list | None}`` – the latter holds the flattened
            Lyapunov matrix if the system is stable.
        """
        A = np.asarray(problem["A"], dtype=float)
        n = A.shape[0]

        # Decision variable: symmetric P
        P = cp.Variable((n, n), PSD=True)

        # Constraints
        # P >> I  (positive definite)
        # A^T P A - P << -I  (Lyapunov inequality)
        constraints: List[cp.Expression] = [
            P >> np.eye(n),
            A.T @ P @ A - P << -np.eye(n)
        ]

        # Solve: objective is zero; we only need feasibility
        prob = cp.Problem(cp.Minimize(0), constraints)

        # The SCS solver is fast for feasibility problems.  
        # Disable verbosity to keep the kernel quiet.
        prob.solve(solver=cp.SCS, verbose=False, eps=1e-7, max_iters=5000)

        if prob.status in {"optimal", "optimal_inaccurate"}:
            return {"is_stable": True, "P": P.value.tolist()}
        # Any other status -> not stable or solver failed
        return {"is_stable": False, "P": None}