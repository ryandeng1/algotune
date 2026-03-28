from typing import Any
import cvxpy as cp
import numpy as np

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        n = problem["num_conditions"]
        cond = problem["conditions"]

        A = cp.Variable(pos=True)
        S = cp.Variable(pos=True)
        V = cp.Variable(n, pos=True)
        W = cp.Variable(n, pos=True)
        Re = cp.Variable(n, pos=True)
        C_D = cp.Variable(n, pos=True)
        C_L = cp.Variable(n, pos=True)
        C_f = cp.Variable(n, pos=True)
        W_w = cp.Variable(n, pos=True)

        constraints = []
        drag = 0
        for i in range(n):
            C = cond[i]
            CDA0 = C["CDA0"]; C_Lmax = C["C_Lmax"]; N_ult = C["N_ult"]
            S_wetratio = C["S_wetratio"]; V_min = C["V_min"]
            W_0 = C["W_0"]; w1 = C["W_W_coeff1"]; w2 = C["W_W_coeff2"]
            e = C["e"]; k = C["k"]; mu = C["mu"]; rho = C["rho"]; tau = C["tau"]

            drag += 0.5 * rho * V[i]**2 * C_D[i] * S

            constraints += [
                C_D[i] >= CDA0 / S + k * C_f[i] * S_wetratio + C_L[i]**2 / (np.pi * A * e),
                C_f[i] >= 0.074 / Re[i]**0.2,
                Re[i] * mu >= rho * V[i] * cp.sqrt(S / A),
                W_w[i] >= w2 * S + w1 * N_ult * A**1.5 * cp.sqrt(W_0 * W[i]) / tau,
                W[i] >= W_0 + W_w[i],
                W[i] <= 0.5 * rho * V[i]**2 * C_L[i] * S,
                2 * W[i] / (rho * V_min**2 * S) <= C_Lmax
            ]

        prob = cp.Problem(cp.Minimize(drag / n), constraints)
        try:
            prob.solve(gp=True)
        except Exception:
            return {"A": [], "S": [], "avg_drag": 0.0, "condition_results": []}

        if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or A.value is None:
            return {"A": [], "S": [], "avg_drag": 0.0, "condition_results": []}

        results = []
        for i in range(n):
            c = cond[i]
            results.append({
                "condition_id": c["condition_id"],
                "V": float(V[i].value),
                "W": float(W[i].value),
                "W_w": float(W_w[i].value),
                "C_L": float(C_L[i].value),
                "C_D": float(C_D[i].value),
                "C_f": float(C_f[i].value),
                "Re": float(Re[i].value),
                "drag": float(0.5 * c["rho"] * V[i].value**2 * C_D[i].value * S.value)
            })

        return {
            "A": float(A.value),
            "S": float(S.value),
            "avg_drag": float(np.mean([r["drag"] for r in results])),
            "condition_results": results
        }