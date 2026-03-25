# solver.py
from typing import Any, Dict, List
import cvxpy as cp
import numpy as np


class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Solve the aircraft wing design optimization problem using CVXPY.

        Parameters
        ----------
        problem : dict
            Problem parameters as described in the task.

        Returns
        -------
        dict
            Dictionary containing the optimal aspect ratio `A`, wing area `S`,
            average drag `avg_drag` and a list of condition results.
        """
        # Extract problem parameters
        num_conditions = problem["num_conditions"]
        conditions = problem["conditions"]

        # Shared design variables
        A = cp.Variable(pos=True, name="A")
        S = cp.Variable(pos=True, name="S")

        # Per-condition variables
        V = [cp.Variable(pos=True, name=f"V_{i}") for i in range(num_conditions)]
        W = [cp.Variable(pos=True, name=f"W_{i}") for i in range(num_conditions)]
        Re = [cp.Variable(pos=True, name=f"Re_{i}") for i in range(num_conditions)]
        C_D = [cp.Variable(pos=True, name=f"C_D_{i}") for i in range(num_conditions)]
        C_L = [cp.Variable(pos=True, name=f"C_L_{i}") for i in range(num_conditions)]
        C_f = [cp.Variable(pos=True, name=f"C_f_{i}") for i in range(num_conditions)]
        W_w = [cp.Variable(pos=True, name=f"W_w_{i}") for i in range(num_conditions)]

        constraints = []
        total_drag = 0

        for i in range(num_conditions):
            cond = conditions[i]
            CDA0 = float(cond["CDA0"])
            C_Lmax = float(cond["C_Lmax"])
            N_ult = float(cond["N_ult"])
            S_wetratio = float(cond["S_wetratio"])
            V_min = float(cond["V_min"])
            W_0 = float(cond["W_0"])
            W_W_coeff1 = float(cond["W_W_coeff1"])
            W_W_coeff2 = float(cond["W_W_coeff2"])
            e = float(cond["e"])
            k = float(cond["k"])
            mu = float(cond["mu"])
            rho = float(cond["rho"])
            tau = float(cond["tau"])

            # Drag for condition
            drag_i = 0.5 * rho * V[i] ** 2 * C_D[i] * S
            total_drag += drag_i

            # Constraints
            constraints += [
                C_D[i] >= CDA0 / S + k * C_f[i] * S_wetratio + C_L[i] ** 2 / (np.pi * A * e),
                C_f[i] >= 0.074 / Re[i] ** 0.2,
                Re[i] * mu >= rho * V[i] * cp.sqrt(S / A),
                W_w[i] >= W_W_coeff2 * S + W_W_coeff1 * N_ult * (A**1.5) * cp.sqrt(W_0 * W[i]) / tau,
                W[i] >= W_0 + W_w[i],
                W[i] <= 0.5 * rho * V[i] ** 2 * C_L[i] * S,
                2 * W[i] / (rho * V_min ** 2 * S) <= C_Lmax,
            ]

        objective = cp.Minimize(total_drag / num_conditions)

        prob = cp.Problem(objective, constraints)

        try:
            prob.solve(gp=True, eps=1e-8, max_iters=2000, verbose=False)
        except Exception:
            # Return empty indicators on failure
            return {"A": [], "S": [], "avg_drag": 0.0, "condition_results": []}

        if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or A.value is None:
            return {"A": [], "S": [], "avg_drag": 0.0, "condition_results": []}

        # Build results
        condition_results: List[Dict[str, Any]] = []
        for i in range(num_conditions):
            cond = conditions[i]
            condition_results.append(
                {
                    "condition_id": cond["condition_id"],
                    "V": float(V[i].value),
                    "W": float(W[i].value),
                    "W_w": float(W_w[i].value),
                    "C_L": float(C_L[i].value),
                    "C_D": float(C_D[i].value),
                    "C_f": float(C_f[i].value),
                    "Re": float(Re[i].value),
                    "drag": float(
                        0.5
                        * cond["rho"]
                        * V[i].value ** 2
                        * C_D[i].value
                        * S.value
                    ),
                }
            )

        return {
            "A": float(A.value),
            "S": float(S.value),
            "avg_drag": float(prob.value),
            "condition_results": condition_results,
        }
