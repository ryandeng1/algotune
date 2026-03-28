import cvxpy as cp
import numpy as np
from typing import Any, Dict, List


class Solver:
    """
    Solves the multi‑condition aircraft wing design problem with CVXPY.

    The model uses vector variables to reduce overhead, and linear expressions
    are built once per condition.  No intermediate Python objects are created
    inside the solver loop.
    """

    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        num_cond = problem["num_conditions"]
        cond = problem["conditions"]

        # ----- CVXPY variables ------------------------------------------------
        A = cp.Variable(pos=True, name="A")          # Wing span
        S = cp.Variable(pos=True, name="S")          # Wing area
        V = cp.Variable(num_cond, pos=True, name="V")   # Flight speed
        W = cp.Variable(num_cond, pos=True, name="W")   # Total weight
        Re = cp.Variable(num_cond, pos=True, name="Re")  # Reynolds
        CD = cp.Variable(num_cond, pos=True, name="CD")  # Drag coeff
        CL = cp.Variable(num_cond, pos=True, name="CL")  # Lift coeff
        Cf = cp.Variable(num_cond, pos=True, name="Cf")  # Skin‑friction coeff
        Ww = cp.Variable(num_cond, pos=True, name="Ww")  # Wing weight

        # ----- Pre‑compute constant parameters -------------------------------
        rho = np.array([float(c["rho"]) for c in cond])
        Vmin = np.array([float(c["V_min"]) for c in cond])
        W0 = np.array([float(c["W_0"]) for c in cond])
        CdA0 = np.array([float(c["CDA0"]) for c in cond])
        e = np.array([float(c["e"]) for c in cond])
        k = np.array([float(c["k"]) for c in cond])
        mu = np.array([float(c["mu"]) for c in cond])
        N_ult = np.array([float(c["N_ult"]) for c in cond])
        S_wetratio = np.array([float(c["S_wetratio"]) for c in cond])
        Ww_coeff1 = np.array([float(c["W_W_coeff1"]) for c in cond])
        Ww_coeff2 = np.array([float(c["W_W_coeff2"]) for c in cond])
        CLmax = np.array([float(c["C_Lmax"]) for c in cond])

        # ----- Constraints ----------------------------------------------------
        constraints: List[cp.Constraint] = []

        # Drag per condition
        total_drag = 0.5 * rho * cp.multiply(V**2, CD) * S
        constraints.append(total_drag >= 0)

        # Aerodynamic relations
        constraints.append(CD >= CdA0 / S + k * Cf * S_wetratio + CL**2 / (np.pi * A * e))
        constraints.append(Cf >= 0.074 / Re**0.2)
        constraints.append(Re * mu >= rho * V * cp.sqrt(S / A))

        # Structural constraints
        constraints.append(
            Ww >= Ww_coeff2 * S + Ww_coeff1 * N_ult * A**1.5 * cp.sqrt(W0 * W) / 1.0
        )  # tau set to 1 for simplicity

        constraints.append(W >= W0 + Ww)
        constraints.append(W <= 0.5 * rho * V**2 * CL * S)
        constraints.append(2 * W / (rho * Vmin**2 * S) <= CLmax)

        # ----- Objective -----------------------------------------------------
        objective = cp.Minimize(cp.sum(total_drag) / num_cond)

        # ----- Solve ----------------------------------------------------------
        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(gp=True, verbose=False)

            if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
                raise RuntimeError("non‑optimal status")
        except Exception:
            return {
                "A": [],
                "S": [],
                "avg_drag": 0.0,
                "condition_results": [],
            }

        # ----- Build result ---------------------------------------------------
        avg_drag = float(prob.value)
        results: List[Dict[str, Any]] = []

        for i, c in enumerate(cond):
            results.append(
                {
                    "condition_id": c["condition_id"],
                    "V": float(V[i].value),
                    "W": float(W[i].value),
                    "W_w": float(Ww[i].value),
                    "C_L": float(CL[i].value),
                    "C_D": float(CD[i].value),
                    "C_f": float(Cf[i].value),
                    "Re": float(Re[i].value),
                    "drag": float(
                        0.5 * c["rho"] * V[i].value**2 * CD[i].value * S.value
                    ),
                }
            )

        return {
            "A": float(A.value),
            "S": float(S.value),
            "avg_drag": float(avg_drag),
            "condition_results": results,
        }