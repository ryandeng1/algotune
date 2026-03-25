import cvxpy as cp
import numpy as np

class Solver:
    def solve(self, problem: dict) -> dict:
        P = np.array(problem["P"])
        R = np.array(problem["R"])
        B = np.array(problem["B"])
        c = np.array(problem["c"])
        T = np.array(problem["T"])
        m, n = P.shape
        
        D = cp.Variable((m, n))
        
        revenue_per_ad = [cp.minimum(R[i] * (P[i] @ D[i]), B[i]) for i in range(m)]
        total_revenue = cp.sum(revenue_per_ad)
        
        constraints = [
            D >= 0,
            cp.sum(D, axis=0) <= T,
            cp.sum(D, axis=1) >= c
        ]
        
        prob = cp.Problem(cp.Maximize(total_revenue), constraints)
        prob.solve(solver='scs')
        
        if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            return {"status": prob.status, "optimal": False}
        
        D_val = D.value
        clicks = np.zeros(m)
        revenue = np.zeros(m)
        
        for i in range(m):
            clicks[i] = np.sum(P[i] * D_val[i])
            revenue[i] = min(R[i] * clicks[i], B[i])
        
        return {
            "status": prob.status,
            "optimal": True,
            "displays": D_val.tolist(),
            "clicks": clicks.tolist(),
            "revenue_per_ad": revenue.tolist(),
            "total_revenue": float(np.sum(revenue)),
            "objective_value": float(prob.value),
        }
