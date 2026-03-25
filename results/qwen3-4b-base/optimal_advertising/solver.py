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
        
        D = cp.Variable((m, n), nonneg=True)
        y = cp.Variable(m)
        
        constraints = [
            D >= 0,
            cp.sum(D, axis=0) <= T,
            cp.sum(D, axis=1) >= c
        ]
        
        for i in range(m):
            constraints.append(y[i] <= R[i] * (P[i, :] @ D[i, :]))
            constraints.append(y[i] <= B[i])
            constraints.append(y[i] >= 0)
        
        objective = cp.Maximize(cp.sum(y))
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.ECOS)
        
        if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            return {"status": prob.status, "optimal": False}
        
        D_val = D.value
        clicks = np.zeros(m)
        revenue_per_ad = np.zeros(m)
        
        for i in range(m):
            clicks[i] = np.sum(P[i, :] * D_val[i, :])
            revenue_per_ad[i] = min(R[i] * clicks[i], B[i])
        
        total_revenue = np.sum(revenue_per_ad)
        
        return {
            "status": prob.status,
            "optimal": True,
            "displays": D_val.tolist(),
            "clicks": clicks.tolist(),
            "revenue_per_ad": revenue_per_ad.tolist(),
            "total_revenue": float(total_revenue)
        }
