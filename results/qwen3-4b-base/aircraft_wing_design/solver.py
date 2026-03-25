import cvxpy as cp
import numpy as np

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        num_conditions = problem["num_conditions"]
        conditions = problem["conditions"]
        
        A = cp.Variable(pos=True)
        S = cp.Variable(pos=True)
        
        V = [cp.Variable(pos=True) for _ in range(num_conditions)]
        W = [cp.Variable(pos=True) for _ in range(num_conditions)]
        C_D = [cp.Variable(pos=True) for _ in range(num_conditions)]
        C_L = [cp.Variable(pos=True) for _ in range(num_conditions)]
        C_f = [cp.Variable(pos=True) for _ in range(num_conditions)]
        W_w = [cp.Variable(pos=True) for _ in range(num_conditions)]
        
        constraints = []
        total_drag = 0.0
        
        for i in range(num_conditions):
            condition = conditions[i]
            CDA0 = float(condition["CDA0"])
            C_Lmax = float(condition["C_Lmax"])
            N_ult = float(condition["N_ult"])
            S_wetratio = float(condition["S_wetratio"])
            V_min = float(condition["V_min"])
            W_0 = float(condition["W_0"])
            W_W_coeff1 = float(condition["W_W_coeff1"])
            W_W_coeff2 = float(condition["W_W_coeff2"])
            e = float(condition["e"])
            k = float(condition["k"])
            mu = float(condition["mu"])
            rho = float(condition["rho"])
            tau = float(condition["tau"])
            
            drag_i = 0.5 * rho * V[i] ** 2 * C_D[i] * S
            total_drag += drag_i
            
            constraints.append(
                C_f[i] >= 0.074 * (mu / (rho * V[i] * cp.sqrt(S / A))) ** 0.2
            )
            
            constraints.append(
                C_D[i] >= CDA0 / S + k * C_f[i] * S_wetratio + C_L[i] ** 2 / (np.pi * A * e)
            )
            
            constraints.append(
                W_w[i] >= W_W_coeff2 * S + W_W_coeff1 * N_ult * (A ** (3 / 2)) * cp.sqrt(W_0 * W[i]) / tau
            )
            
            constraints.append(W[i] >= W_0 + W_w[i])
            
            constraints.append(
                W[i] <= 0.5 * rho * V[i] ** 2 * C_L[i] * S
            )
            
            constraints.append(
                2 * W[i] / (rho * V_min ** 2 * S) <= C_Lmax
            )
        
        objective = cp.Minimize(total_drag / num_conditions)
        prob = cp.Problem(objective, constraints)
        
        try:
            prob.solve(gp=True)
            
            if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or A.value is None:
                return {"A": [], "S": [], "avg_drag": 0.0, "condition_results": []}
            
            condition_results = []
            for i in range(num_conditions):
                condition_results.append({
                    "condition_id": conditions[i]["condition_id"],
                    "V": float(V[i].value),
                    "W": float(W[i].value),
                    "W_w": float(W_w[i].value),
                    "C_L": float(C_L[i].value),
                    "C_D": float(C_D[i].value),
                    "C_f": float(C_f[i].value),
                    "Re": float((conditions[i]["rho"] * V[i].value * np.sqrt(S.value / A.value)) / conditions[i]["mu"]),
                    "drag": float(0.5 * conditions[i]["rho"] * V[i].value ** 2 * C_D[i].value * S.value)
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
