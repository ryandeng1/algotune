import cvxpy as cp
import numpy as np
from typing import Any, Dict, List


class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Solve a multi‑point aircraft wing design problem using CVXPY.
        The function is heavily micro‑optimised for speed on Python 3.10.

        :param problem: Dictionary with problem parameters
        :return: Dictionary with optimal design variables and results per condition
        """
        # ---------- Input extraction ----------
        num = int(problem["num_conditions"])
        conds = problem["conditions"]

        # ---------- Variables ----------
        A = cp.Variable(name="A", pos=True)          # aspect ratio
        S = cp.Variable(name="S", pos=True)          # wing area

        V = [cp.Variable(name=f"V_{i}", pos=True) for i in range(num)]
        W = [cp.Variable(name=f"W_{i}", pos=True) for i in range(num)]
        Re = [cp.Variable(name=f"Re_{i}", pos=True) for i in range(num)]
        C_D = [cp.Variable(name=f"C_D_{i}", pos=True) for i in range(num)]
        C_L = [cp.Variable(name=f"C_L_{i}", pos=True) for i in range(num)]
        C_f = [cp.Variable(name=f"C_f_{i}", pos=True) for i in range(num)]
        W_w = [cp.Variable(name=f"W_w_{i}", pos=True) for i in range(num)]

        # ---------- Pre‑compute constants ----------
        pi_over_e = np.pi  # will be divided by e later
        coef_Re_mu = np.array(
            [float(cond["mu"]) for cond in conds]
        )  # vector of mu for each condition
        rho_arr = np.array([float(cond["rho"]) for cond in conds])
        V_min_arr = np.array([float(cond["V_min"]) for cond in conds])
        W_0_arr = np.array([float(cond["W_0"]) for cond in conds])
        k_arr = np.array([float(cond["k"]) for cond in conds])
        e_arr = np.array([float(cond["e"]) for cond in conds])
        N_ult_arr = np.array([float(cond["N_ult"]) for cond in conds])
        S_wet_arr = np.array([float(cond["S_wetratio"]) for cond in conds])
        CDA0_arr = np.array([float(cond["CDA0"]) for cond in conds])
        C_Lmax_arr = np.array([float(cond["C_Lmax"]) for cond in conds])
        W_Wc1 = np.array([float(cond["W_W_coeff1"]) for cond in conds])
        W_Wc2 = np.array([float(cond["W_W_coeff2"]) for cond in conds])
        tau_arr = np.array([float(cond["tau"]) for cond in conds])

        # ---------- Constraints & objective ----------
        constraints: List[cp.Constraint] = []
        total_drag = 0

        for i in range(num):
            # Drag term
            total_drag += 0.5 * rho_arr[i] * V[i] ** 2 * C_D[i] * S

            # Drag coefficient model
            constraints.append(
                C_D[i]
                >= CDA0_arr[i] / S
                + k_arr[i] * C_f[i] * S_wet_arr[i]
                + C_L[i] ** 2 / (pi_over_e * A * e_arr[i])
            )

            # Skin friction model
            constraints.append(C_f[i] >= 0.074 / Re[i] ** 0.2)

            # Reynolds number definition
            constraints.append(Re[i] * coef_Re_mu[i] >= rho_arr[i] * V[i] * cp.sqrt(S / A))

            # Wing weight model
            constraints.append(
                W_w[i]
                >= W_Wc2[i] * S
                + W_Wc1[i]
                * N_ult_arr[i]
                * A ** 1.5
                * cp.sqrt(W_0_arr[i] * W[i])
                / tau_arr[i]
            )

            # Total weight constraints
            constraints.append(W[i] >= W_0_arr[i] + W_w[i])
            constraints.append(W[i] <= 0.5 * rho_arr[i] * V[i] ** 2 * C_L[i] * S)
            constraints.append(
                2 * W[i]
                <= rho_arr[i] * V_min_arr[i] ** 2 * S * C_Lmax_arr[i]
            )

        objective = cp.Minimize(total_drag / num)

        # ---------- Solve ----------
        try:
            prob = cp.Problem(objective, constraints)
            prob.solve(gp=True)  # geometric program

            if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or A.value is None:
                return {"A": [], "S": [], "avg_drag": 0.0, "condition_results": []}

            # Build result list
            condition_results = []
            for i, cond in enumerate(conds):
                recipe = {
                    "condition_id": cond["condition_id"],
                    "V": float(V[i].value),
                    "W": float(W[i].value),
                    "W_w": float(W_w[i].value),
                    "C_L": float(C_L[i].value),
                    "C_D": float(C_D[i].value),
                    "C_f": float(C_f[i].value),
                    "Re": float(Re[i].value),
                }
                recipe["drag"] = (
                    0.5 * rho_arr[i] * recipe["V"] ** 2 * recipe["C_D"] * float(S.value)
                )
                condition_results.append(recipe)

            return {
                "A": float(A.value),
                "S": float(S.value),
                "avg_drag": float(prob.value),
                "condition_results": condition_results,
            }

        except Exception:
            return {"A": [], "S": [], "avg_drag": 0.0, "condition_results": []}