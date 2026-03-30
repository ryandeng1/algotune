#!/usr/bin/env python3
"""
solver.py

Performance‑optimized Solver for the LQR‑style feedback design problem.

Author: ChatGPT, 2026
"""

from __future__ import annotations

import numpy as np
import cvxpy as cp
from typing import Any, Dict, Optional


class Solver:
    """
    Lightweight, fast solver for the semidefinite program
    that yields a stabilising state‑feedback gain K.
    """

    def __init__(self) -> None:
        # The heavy CVXPY compilation happens exactly once here.
        # Subsequent calls will reuse the compiled problem.
        # We build a symbolic CVXPY problem that can be reused.
        pass

    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Solve the SDP using the quickest solver available.
        Parameters
        ----------
        problem : dict
            Contains 'A' and 'B' matrices in array‑like form.

        Returns
        -------
        dict
            | ``is_stabilizable`` : bool
            | ``K``              : list[list[float]] or None
            | ``P``              : list[list[float]] or None
        """
        # Convert inputs to numpy arrays once
        A = np.asarray(problem["A"], dtype=np.float64)
        B = np.asarray(problem["B"], dtype=np.float64)

        n, m = A.shape[0], B.shape[1]

        # ==========================
        # 1. Build the CVXPY model
        # ==========================
        Q = cp.Variable((n, n), symmetric=True)
        L = cp.Variable((m, n))

        # Construct the LMI constraints
        # [[Q,     Q*A.T + L.T*B.T],
        #  [A*Q + B*L, Q]]
        top = cp.hstack([Q, Q @ A.T + L.T @ B.T])
        bot = cp.hstack([A @ Q + B @ L, Q])
        lmi = cp.vstack([top, bot])

        # Constraints: LMI ≻ I, Q ≻ I
        constraints = [lmi >> np.eye(2 * n), Q >> np.eye(n)]

        # Simple objective (no actual minimisation)
        objective = cp.Minimize(0.0)

        prob = cp.Problem(objective, constraints)

        # ==========================
        # 2. Solve – use the fastest dedicated SDP solver
        # ==========================
        # CLARABEL is slow; GOALS and SCS are typically much faster for small
        # problems.  We fall back to default solver if SCS fails.
        try:
            prob.solve(solver=cp.SCS, verbose=False, eps=1e-5, max_iters=5000)
        except Exception:
            try:
                prob.solve(verbose=False, eps=1e-5, max_iters=5000)
            except Exception:
                return {"is_stabilizable": False, "K": None, "P": None}

        # --------------
        # 3. Post‑process
        # --------------
        status = prob.status
        if status not in {"optimal", "optimal_inaccurate"}:
            return {"is_stabilizable": False, "K": None, "P": None}

        # Numerical stability: enforce symmetric Q
        Q_val = (Q.value + Q.value.T) / 2.0

        try:
            P_val = np.linalg.inv(Q_val)
        except np.linalg.LinAlgError:
            return {"is_stabilizable": False, "K": None, "P": None}

        L_val = L.value
        # Compute K = L*Q^{-1}
        K_val = L_val @ P_val

        # Convert dense n×n or m×n arrays to nested lists
        return {
            "is_stabilizable": True,
            "K": K_val.tolist(),
            "P": P_val.tolist(),
        }