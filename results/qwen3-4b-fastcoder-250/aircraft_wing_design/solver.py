import cvxpy as cp
import numpy as np

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        num_conditions = problem["num_conditions"]
        conditions = problem["conditions"]
        
        cond_data = []
        for cond in conditions:
            cond_data.append({
                "CDA0": float(cond["CDA0"]),
                "C_Lmax": float(cond["C_Lmax"]),
                "N_ult": float(cond["N_ult"]),
                "S_wetratio": float(cond["S_wetratio"]),
                "V_min": float(cond["V_min"]),
                "W_0": float(cond["W_0"]),
                "W_W_coeff1": float(cond["W_W_coeff1"]),
                "W_W_coeff2": float(cond["W_W_coeff2"]),
                "e": float(cond["e"]),
                "k": float(cond["k"]),
                "mu": float(cond["mu"]),
                "rho": float(cond["rho"]),
                "tau": float(cond["tau"])
            })
        
        A = cp.Variable(pos=True)
        S = cp.Variable(pos=True)
        V = cp.Variable(num_conditions, pos=True)
        W = cp.Variable(num_conditions, pos=True)
        Re = cp.Variable(num_conditions, pos=True)
        C_D = cp.Variable(num_conditions, pos=True)
        C_L = cp.Variable(num_conditions, pos=True)
        C_f = cp.Variable(num_conditions, pos=True)
        W_w = cp.Variable(num_conditions, pos=True)
        
        constraints = []
        total_drag = 0.0
        for i in range(num_conditions):
            cond = cond_data[i]
            total_drag += 0.5 * cond["rho"] * V[i] ** 2 * C_D[i] * S
        
        objective = cp.Minimize(total_drag / num_conditions)
        
        for i in range(num_conditions):
            cond = cond_data[i]
            constraints.append(
                C_D[i] >= cond["CDA0"] / S + cond["k"] * C_f[i] * cond["S_wetratio"] + 
                (C_L[i] ** 2) / (np.pi * A * cond["e"])
            )
            constraints.append(C_f[i] >= 0.074 / Re[i] ** 0.2)
            constraints.append(
                Re[i] * cond["mu"] >= cond["rho"] * V[i] * cp.sqrt(S / A)
            )
            constraints.append(
                W_w[i] >= cond["W_W_coeff2"] * S + 
                cond["W_W_coeff1"] * cond["N_ult"] * (A ** (3 / 2)) * 
                cp.sqrt(cond["W_0"] * W[i]) / cond["tau"]
            )
            constraints.append(W[i] >= cond["W_0"] + W_w[i])
            constraints.append(W[i] <= 0.5 * cond["rho"] * V[i] ** 2 * C_L[i] * S)
            constraints.append(
                2 * W[i] / (cond["rho"] * cond["V_min"] ** 2 * S) <= cond["C_Lmax"]
            )
        
        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(gp=True)
            if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or A.value is None:
                return {"A": [], "S": [], "avg_drag": 0.0, "condition_results": []}
            
            condition_results = []
            for i in range(num_conditions):
                cond = cond_data[i]
                condition_results.append({
                    "condition_id": conditions[i]["condition_id"],
                    "V": float(V[i].value),
                    "W": float(W[i].value),
                    "W_w": float(W_w[i].value),
                    "C_L": float(C_L[i].value),
                    "C_D": float(C_D[i].value),
                    "C_f": float(C_f[i].value),
                    "Re": float(Re[i].value),
                    "drag": 0.5 * cond["rho"] * V[i].value ** 2 * C_D[i].value * S.value
                })
            
            return {
                "A": float(A.value),
                "S": float(S.value),
                "avg_drag": float(prob.value),
                "condition_results": condition_results
            }
        except cp.SolverError:
            return {"A": [], "S": [], "avg_drag": 0.0, "condition_results": []}
        except Exception:
            return {"A": [], "S": [], "avg_drag": 0.0, "condition_results": []}
