# solver.py
"""
Highly‑optimised solver for the SDP based feedback controller design
"""

from __future__ import annotations

import numpy as np
import cvxpy as cp
from typing import Any, Dict


class Solver:
    """
    Solve a controller design LMI:
        | Q            Q Aᵀ + Lᵀ Bᵀ |
        | A Q + B L    Q             |  >  I_2n
        Q > I_n

    Returns the feedback gain K = L Q⁻¹ and the Lyapunov matrix P = Q⁻¹
    if the problem is feasible.
    """

    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        # Extract matrices
        A = np.asarray(problem["A"], dtype=np.float64)
        B = np.asarray(problem["B"], dtype=np.float64)
        n, m = A.shape[0], B.shape[1]

        # SDP variables
        Q = cp.Variable((n, n), symmetric=True)
        L = cp.Variable((m, n))

        # Build constraints
        #   [ Q ,       Q A.T + L.T B.T ]
        #   [ A Q + B L ,           Q   ]  >>  I_2n
        top = cp.hstack([Q, Q @ A.T + L.T @ B.T])
        bottom = cp.hstack([A @ Q + B @ L, Q])
        blk = cp.vstack([top, bottom])

        constraints = [
            blk >> np.eye(2 * n, dtype=np.float64),
            Q >> np.eye(n, dtype=np.float64),
        ]

        # Problem definition
        prob = cp.Problem(cp.Minimize(0), constraints)

        # Solve – use SCS for speed; fall back to ECOS if needed
        try:
            prob.solve(solver=cp.SCS, verbose=False, eps=1e-6, max_iters=2000)
            status = prob.status
        except Exception:
            # If SCS fails, try ECOS
            prob.solve(solver=cp.ECOS, verbose=False)
            status = prob.status

        # Check feasibility
        if status in {"optimal", "optimal_inaccurate"}:
            Q_val = Q.value
            L_val = L.value

            # Compute K = L Q⁻¹ using solve for numerical stability
            K_val = np.linalg.solve(Q_val.T, L_val.T).T  # equivalent to L @ Q⁻¹

            # Lyapunov matrix
            P_val = np.linalg.inv(Q_val)

            return {
                "is_stabilizable": True,
                "K": K_val.tolist(),
                "P": P_val.tolist(),
            }

        return {"is_stabilizable": False, "K": None, "P": None}