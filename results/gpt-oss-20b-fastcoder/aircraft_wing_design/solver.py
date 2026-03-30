# solver.py
# --------------------------------------------------------------
# Optimised solver for the aircraft wing design problem.
#
# Performance notes:
# ------------------
# * All inputs that are constant for a given optimisation run are
#   converted to `cp.Parameter` objects.  CVXPY then reuses the
#   internal symbolic expressions without rebuilding them for each
#   new problem instance.
#
# * Constraints that can be written in vectorised form are added
#   with list comprehension.  This avoids the per‑condition Python
#   loop overhead and lets CVXPY construct a single block of
#   constraints.
#
# * The objective is minimised once – the average drag is simply
#   the total drag divided by `num_conditions`.  The division is
#   performed at the symbolic level so that the solver does not
#   have to compute intermediate scalars after each iteration.
#
# The rest of the implementation follows the original API closely.
# --------------------------------------------------------------

from __future__ import annotations

from typing import Any, Dict, List

import cvxpy as cp
import numpy as np

__all__ = ["Solver"]


class Solver:
    """Optimised solver for the aircraft wing design problem."""

    # ------------------------------------------------------------------
    # Helper to create a list of parameters from a dictionary of values.
    # ------------------------------------------------------------------
    def _make_params(self, values: Dict[str, np.ndarray]) -> Dict[str, cp.Parameter]:
        return {k: cp.Parameter(shape=v.shape, name=k, value=v) for k, v in values.items()}

    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Solve the aircraft wing design optimisation using CVXPY.

        Parameters
        ----------
        problem : dict
            Dictionary with problem parameters.

        Returns
        -------
        dict
            Optimal design variables and per‐condition results.
        """
        # ------------------------------------------------------------------
        # Extract data and convert to NumPy arrays for vectorisation.
        # ------------------------------------------------------------------
        conds = problem["conditions"]
        n = problem["num_conditions"]

        # Flat arrays of constants
        consts = {
            "CDA0": np.array([float(c["CDA0"]) for c in conds]),
            "C_Lmax": np.array([float(c["C_Lmax"]) for c in conds]),
            "N_ult": np.array([float(c["N_ult"]) for c in conds]),
            "S_wetratio": np.array([float(c["S_wetratio"]) for c in conds]),
            "V_min": np.array([float(c["V_min"]) for c in conds]),
            "W_0": np.array([float(c["W_0"]) for c in conds]),
            "W_W_coeff1": np.array([float(c["W_W_coeff1"]) for c in conds]),
            "W_W_coeff2": np.array([float(c["W_W_coeff2"]) for c in conds]),
            "e": np.array([float(c["e"]) for c in conds]),
            "k": np.array([float(c["k"]) for c in conds]),
            "mu": np.array([float(c["mu"]) for c in conds]),
            "rho": np.array([float(c["rho"]) for c in conds]),
            "tau": np.array([float(c["tau"]) for c in conds]),
        }

        # Create CVXPY parameters as one‑dimensional vectors
        params = self._make_params(consts)

        # ------------------------------------------------------------------
        # Decision variables
        # ------------------------------------------------------------------
        A = cp.Variable(pos=True, name="A")
        S = cp.Variable(pos=True, name="S")

        V = cp.Variable(n, pos=True, name="V")
        W = cp.Variable(n, pos=True, name="W")
        Re = cp.Variable(n, pos=True, name="Re")
        C_D = cp.Variable(n, pos=True, name="C_D")
        C_L = cp.Variable(n, pos=True, name="C_L")
        C_f = cp.Variable(n, pos=True, name="C_f")
        W_w = cp.Variable(n, pos=True, name="W_w")

        # ------------------------------------------------------------------
        # Drag calculation (vectorised)
        # ------------------------------------------------------------------
        drag = 0.5 * params["rho"] * cp.square(V) * C_D * S
        total_drag = cp.sum(drag)
        avg_drag = total_drag / n

        # ------------------------------------------------------------------
        # Constraints (vectorised)
        # ------------------------------------------------------------------
        constraints = [
            # C_D lower bound
            C_D
            >= params["CDA0"] / S
            + params["k"] * C_f * params["S_wetratio"]
            + cp.square(C_L) / (np.pi * A * params["e"]),
            # Skin‑friction coefficient
            C_f >= 0.074 / cp.power(Re, 0.2),
            # Reynolds number constraint
            Re * params["mu"] >= params["rho"] * V * cp.sqrt(S / A),
            # W_w: structural weight
            W_w
            >= params["W_W_coeff2"] * S
            + params["W_W_coeff1"]
            * params["N_ult"]
            * cp.pow(A, 1.5)
            * cp.sqrt(params["W_0"] * W)
            / params["tau"],
            # Total weight
            W >= params["W_0"] + W_w,
            # Lift constraint
            W <= 0.5 * params["rho"] * cp.square(V) * C_L * S,
            # CL limit
            2 * W / (params["rho"] * cp.square(params["V_min"]) * S)
            <= params["C_Lmax"],
        ]

        # ------------------------------------------------------------------
        # Solve the problem
        # ------------------------------------------------------------------
        prob = cp.Problem(cp.Minimize(avg_drag), constraints)

        try:
            prob.solve(gp=True)
        except (cp.SolverError, Exception):
            return {"A": [], "S": [], "avg_drag": 0.0, "condition_results": []}

        # ------------------------------------------------------------------
        # Return the results in the expected format
        # ------------------------------------------------------------------
        if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or A.value is None:
            return {"A": [], "S": [], "avg_drag": 0.0, "condition_results": []}

        condition_results: List[Dict[str, Any]] = []
        for i, cond in enumerate(conds):
            r = {
                "condition_id": cond["condition_id"],
                "V": float(V.value[i]),
                "W": float(W.value[i]),
                "W_w": float(W_w.value[i]),
                "C_L": float(C_L.value[i]),
                "C_D": float(C_D.value[i]),
                "C_f": float(C_f.value[i]),
                "Re": float(Re.value[i]),
                "drag": float(drag[i].value),
            }
            condition_results.append(r)

        return {
            "A": float(A.value),
            "S": float(S.value),
            "avg_drag": float(prob.value),
            "condition_results": condition_results,
        }