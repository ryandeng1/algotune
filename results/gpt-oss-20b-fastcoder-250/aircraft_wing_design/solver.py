import cvxpy as cp
import numpy as np
from typing import Any, Dict, List

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Solve the aircraft wing design optimisation problem with CVXPY.
        Uses vectorised constraints to reduce the CVXPY model size.
        """
        # Extract parameters
        n = problem["num_conditions"]
        cond = problem["conditions"]

        # Shared design variables
        A = cp.Variable(pos=True, name="A")
        S = cp.Variable(pos=True, name="S")

        # Per‑condition variables
        V = cp.Variable(n, pos=True, name="V")
        W = cp.Variable(n, pos=True, name="W")
        Re = cp.Variable(n, pos=True, name="Re")
        C_D = cp.Variable(n, pos=True, name="C_D")
        C_L = cp.Variable(n, pos=True, name="C_L")
        C_f = cp.Variable(n, pos=True, name="C_f")
        W_w = cp.Variable(n, pos=True, name="W_w")

        # Scalars that are constant across conditions
        rho  = np.array([float(c["rho"]) for c in cond])
        V_min = np.array([float(c["V_min"]) for c in cond])
        C_DA0 = np.array([float(c["CDA0"]) for c in cond])
        C_Lmax = np.array([float(c["C_Lmax"]) for c in cond])
        N_ult = np.array([float(c["N_ult"]) for c in cond])
        S_wet = np.array([float(c["S_wetratio"]) for c in cond])
        W_0 = np.array([float(c["W_0"]) for c in cond])
        Wc1 = np.array([float(c["W_W_coeff1"]) for c in cond])
        Wc2 = np.array([float(c["W_W_coeff2"]) for c in cond])
        e = np.array([float(c["e"]) for c in cond])
        k = np.array([float(c["k"]) for c in cond])
        mu = np.array([float(c["mu"]) for c in cond])

        # Helper multiples for vectorised constraints
        piA = np.pi * A
        sqrtS_A = cp.sqrt(S / A)

        # Constraint matrix
        constraints = []

        # Drag coefficient expression
        constraints.append(
            C_D >= C_DA0 / S + k * C_f * S_wet + C_L**2 / (piA * e)
        )
        constraints.append(C_f >= 0.074 / Re**0.2)
        constraints.append(Re * mu >= rho * V * sqrtS_A)

        # Wing weight
        constraints.append(
            W_w >= Wc2 * S + Wc1 * N_ult * A**1.5 * cp.sqrt(W_0 * W) / 1.0
        )
        constraints.append(W >= W_0 + W_w)

        # Lift equals weight
        constraints.append(W <= 0.5 * rho * V**2 * C_L * S)

        # Stall constraint
        constraints.append(2 * W / (rho * V_min**2 * S) <= C_Lmax)

        # Objective: average drag
        drag = 0.5 * rho * V**2 * C_D * S
        objective = cp.Minimize(cp.sum(drag) / n)

        # Solve using GP
        try:
            prob = cp.Problem(objective, constraints)
            prob.solve(gp=True)

            if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or A.value is None:
                return {"A": [], "S": [], "avg_drag": 0.0, "condition_results": []}

            # Pack results
            res = {
                "A": float(A.value),
                "S": float(S.value),
                "avg_drag": float(prob.value),
                "condition_results": []
            }

            for i in range(n):
                res["condition_results"].append(
                    {
                        "condition_id": cond[i]["condition_id"],
                        "V": float(V[i].value),
                        "W": float(W[i].value),
                        "W_w": float(W_w[i].value),
                        "C_L": float(C_L[i].value),
                        "C_D": float(C_D[i].value),
                        "C_f": float(C_f[i].value),
                        "Re": float(Re[i].value),
                        "drag": 0.5 * rho[i] * V[i].value**2 * C_D[i].value * S.value,
                    }
                )
            return res

        except Exception:
            return {"A": [], "S": [], "avg_drag": 0.0, "condition_results": []}