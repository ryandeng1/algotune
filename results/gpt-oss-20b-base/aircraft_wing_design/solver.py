# solver.py
from __future__ import annotations

import numpy as np

import cvxpy as cp
from typing import Any, Dict, List


class Solver:
    """
    Aircraft wing design solver based on geometric programming (CVXPY).

    >>> solver = Solver()
    >>> problem = {
    ...     "num_conditions": 2,
    ...     "conditions": [
    ...         {"condition_id": 0, "CDA0": 0.027, "C_Lmax": 1.5, "N_ult": 2.5, "S_wetratio": 0.1,
    ...          "V_min": 80, "W_0": 6000, "W_W_coeff1": 0.02, "W_W_coeff2": 0.01,
    ...          "e": 0.8, "k": 1.5, "mu": 1.0e-5, "rho": 0.8, "tau": 0.8, "rho": 0.8},
    ...         {"condition_id": 1, "CDA0": 0.027, "C_Lmax": 1.5, "N_ult": 2.5, "S_wetratio": 0.1,
    ...          "V_min": 80, "W_0": 6000, "W_W_coeff1": 0.02, "W_W_coeff2": 0.01,
    ...          "e": 0.8, "k": 1.5, "mu": 1.0e-5, "rho": 0.8, "tau": 0.8, "rho": 0.8}
    ...     ]
    ... }
    >>> res = solver.solve(problem)
    >>> res["A"]  # Axis ratio
    7.3...          # example value
    """

    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Solve the design problem and return solution metrics."""
        n = int(problem["num_conditions"])
        cond = problem["conditions"]

        # ----- Variables -----
        A = cp.Variable(nonneg=True, name="A")   # aspect ratio
        S = cp.Variable(nonneg=True, name="S")   # wing area

        V = cp.Variable(n, nonneg=True, name="V")
        W = cp.Variable(n, nonneg=True, name="W")
        Re = cp.Variable(n, nonneg=True, name="Re")
        C_D = cp.Variable(n, nonneg=True, name="C_D")
        C_L = cp.Variable(n, nonneg=True, name="C_L")
        C_f = cp.Variable(n, nonneg=True, name="C_f")
        W_w = cp.Variable(n, nonneg=True, name="W_w")

        # ----- Constants as parameters -----
        const = {k: cp.Parameter(nonneg=True, name=k) for k in (
            "CDA0", "C_Lmax", "N_ult", "S_wetratio",
            "V_min", "W_0", "W_W_coeff1", "W_W_coeff2",
            "e", "k", "mu", "rho", "tau"
        )}

        # Set parameter values
        for i, row in enumerate(cond):
            for k in const:
                const[k].value = float(row[k])

        # ----- Constraints -----
        cons: List[cp.Constraint] = []

        # Vectorised formulation reduces Python loop overhead
        CDA0 = const["CDA0"]
        k_val = const["k"]
        e_val = const["e"]

        # Drag related
        cons.append(C_D >= CDA0 / S + k_val * C_f * const["S_wetratio"] + (C_L ** 2) / (np.pi * A * e_val))
        # Skin friction
        cons.append(C_f >= 0.074 / (Re ** 0.2))
        # Reynolds
        cons.append(Re * const["mu"] >= const["rho"] * V * cp.sqrt(S / A))
        # Wing weight
        cons.append(W_w >= const["W_W_coeff2"] * S + const["W_W_coeff1"] * const["N_ult"] * A ** 1.5 *
                   cp.sqrt(const["W_0"] * W) / const["tau"])
        # Total weight
        cons.append(W >= const["W_0"] + W_w)
        # Lift requirement (upper bound)
        cons.append(W <= 0.5 * const["rho"] * V ** 2 * C_L * S)
        # Lift-to-drag ratio limit
        cons.append(2 * W / (const["rho"] * const["V_min"] ** 2 * S) <= const["C_Lmax"])

        # ----- Objective -----
        drag = 0.5 * const["rho"] * V ** 2 * C_D * S
        objective = cp.Minimize(cp.sum(drag) / n)

        prob = cp.Problem(objective, cons)

        # Solve with geometric programming enabled
        try:
            prob.solve(gp=True)
        except cp.SolverError:
            return {"A": [], "S": [], "avg_drag": 0.0, "condition_results": []}
        except Exception:
            return {"A": [], "S": [], "avg_drag": 0.0, "condition_results": []}

        if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or A.value is None:
            return {"A": [], "S": [], "avg_drag": 0.0, "condition_results": []}

        # ----- Pack results -----
        condition_results = []
        for i in range(n):
            drag_val = 0.5 * float(cond[i]["rho"]) * V[i].value ** 2 * C_D[i].value * S.value
            condition_results.append({
                "condition_id": cond[i]["condition_id"],
                "V": float(V[i].value),
                "W": float(W[i].value),
                "W_w": float(W_w[i].value),
                "C_L": float(C_L[i].value),
                "C_D": float(C_D[i].value),
                "C_f": float(C_f[i].value),
                "Re": float(Re[i].value),
                "drag": float(drag_val),
            })

        return {
            "A": float(A.value),
            "S": float(S.value),
            "avg_drag": float(prob.value),
            "condition_results": condition_results,
        }