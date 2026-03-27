#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from scipy.optimize import linprog


class Solver:
    """
    Solves the following linear program:

        minimise   sum(P)
        subject to
            P_min <= P <= P_max
            G[i,i] * P[i] >= S_min * (σ[i] + sum_j G[i,j] * P[j] - G[i,i] * P[i])
            for all i

    The constraints are transformed into standard form A_ub @ P <= b_ub
    and A_eq @ P == b_eq using the high‑performance HiGHS solver.
    """

    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        # ------------------------------------------------------------------
        # Input parsing – all arrays are cast to np.ndarray of float.
        # ------------------------------------------------------------------
        G: np.ndarray = np.asarray(problem["G"], dtype=float)
        sigma: np.ndarray = np.asarray(problem["σ"], dtype=float)
        P_min: np.ndarray = np.asarray(problem["P_min"], dtype=float)
        P_max: np.ndarray = np.asarray(problem["P_max"], dtype=float)
        S_min: float = float(problem["S_min"])

        n: int = G.shape[0]

        # ------------------------------------------------------------------
        # Build the linear inequality constraints A_ub @ P <= b_ub
        # The original constraints are of the form
        #
        #   g_i * P_i >= S_min * (σ_i + (G P)_i - g_i * P_i)
        #
        # Rewriting:
        #
        #   (g_i * (1 + S_min)) * P_i - S_min * (G P)_i >= S_min * σ_i
        #
        # which can be expressed as:
        #
        #   (g_i * (1 + S_min) * e_i^T - S_min * G) @ P >= S_min * σ
        #
        # Multiplying by -1 gives the standard "≤" form.
        # ------------------------------------------------------------------
        diag_g = np.diag(G).astype(float)  # g_i
        factor = diag_g * (1.0 + S_min)  # g_i * (1 + S_min)

        # Coefficient matrix for the inequality constraints
        A_ub = -S_min * G
        A_ub[np.arange(n), np.arange(n)] += factor  # add diagonal part

        b_ub = -S_min * sigma  # move RHS to LHS and flip sign

        # ------------------------------------------------------------------
        # Variable bounds: P_min <= P <= P_max
        # ------------------------------------------------------------------
        bounds: List[tuple[float, float]] = [(float(P_min[i]), float(P_max[i]))
                                             for i in range(n)]

        # ------------------------------------------------------------------
        # Objective: minimise sum(P) -> c = [1, 1, ..., 1]
        # ------------------------------------------------------------------
        c = np.ones(n, dtype=float)

        # ------------------------------------------------------------------
        # Call HiGHS linear programming solver via SciPy
        # ------------------------------------------------------------------
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                      method="highs", options={"presolve": True})

        if res.status not in (0, 2):  # 0: optimal, 2: infeasible
            raise RuntimeError(f"Linear solver failed (status={res.status})")

        return {"P": res.x.tolist(), "objective": float(res.fun)}