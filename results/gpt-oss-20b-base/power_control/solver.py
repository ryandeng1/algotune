# solver.py
from __future__ import annotations

from typing import Any, Dict

import cvxpy as cp
import numpy as np


class Solver:
    """Fast solver for the energy allocation problem.

    The original implementation built the constraints in a Python loop, which
    paid a cost in Python overhead.  This refactor uses fully vectorised
    matrix operations to construct the constraints in a single call,
    dramatically reducing the runtime while keeping the problem formulation
    identical and still using cvxpy’s highly optimised solver backend.
    """

    @staticmethod
    def solve(problem: Dict[str, Any]) -> Dict[str, Any]:
        """Return power schedule `P` that minimises total power subject to
        per‑generator limits and a hydrostatic constraint.
        """
        # Convert all data to NumPy arrays – this costs almost nothing relative
        # to the optimisation routine but guarantees that we work with dense
        # representation once in cvxpy.
        G = np.asarray(problem["G"], dtype=float)
        sigma = np.asarray(problem["σ"], dtype=float)
        P_min = np.asarray(problem["P_min"], dtype=float)
        P_max = np.asarray(problem["P_max"], dtype=float)
        S_min = float(problem["S_min"])

        n = G.shape[0]

        # Decision variable
        P = cp.Variable(n, nonneg=True)

        # base variable bounds (vectorised)
        constraints = [P >= P_min, P <= P_max]

        # --------------------------------------------------------------------
        # Hydrostatic constraint, vectorised:
        #
        #   G[i,i] * P[i]  >=  S_min * ( σ[i] + sum_j G[i,j] P[j] - G[i,i] P[i] )
        #
        #        ⇔  G[i,i] * P[i] + S_min * G[i,i] * P[i]
        #        ≥   S_min * σ[i] + S_min * (G @ P)[i]
        #
        #        ⇔  diag(G) * (1+S_min) * P
        #        ≥   S_min * σ + S_min * (G @ P)
        #
        # Since all terms are linear in *P*, we can express the whole set
        # of inequalities in a single vectorized statement.
        # --------------------------------------------------------------------
        diagonal_G = np.diag(G)
        # left hand side: elementwise multiply of P with (1+S_min)*diag(G)
        lhs = cp.multiply((1.0 + S_min) * diagonal_G, P)
        # right hand side: S_min * sigma + S_min * (G @ P)
        rhs = S_min * sigma + S_min * (G @ P)
        constraints.append(lhs >= rhs)

        # minimise total power
        objective = cp.Minimize(cp.sum(P))
        prob = cp.Problem(objective, constraints)

        # ECOS is fast and stable for this linear program
        prob.solve(solver=cp.ECOS, verbose=False)

        if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise RuntimeError(f"Solver failed: {prob.status}")

        return {"P": P.value.tolist(), "objective": float(prob.value)}