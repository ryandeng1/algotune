import cvxpy as cp
import numpy as np

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        A = np.array(problem["A"])
        B = np.array(problem["B"])
        C = np.array(problem["C"])
        y = np.array(problem["y"])
        x0 = np.array(problem["x_initial"])
        tau = float(problem["tau"])
        M = float(problem["M"])
        
        N, m = y.shape
        n = A.shape[1]
        p = B.shape[1]
        
        x = cp.Variable((N + 1, n))
        w = cp.Variable((N, p))
        v = cp.Variable((N, m))
        
        process_noise = cp.sum_squares(w)
        v_norm = cp.norm(v, axis=1)
        huber_loss = cp.huber(v_norm, M)
        measurement_noise = tau * cp.sum(huber_loss)
        obj = cp.Minimize(process_noise + measurement_noise)
        
        constraints = [x[0] == x0]
        for t in range(N):
            constraints.append(x[t+1] == A @ x[t] + B @ w[t])
            constraints.append(y[t] == C @ x[t] + v[t])
        
        prob = cp.Problem(obj, constraints)
        try:
            prob.solve()
        except Exception:
            return {"x_hat": [], "w_hat": [], "v_hat": []}
        
        if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE] or x.value is None:
            return {"x_hat": [], "w_hat": [], "v_hat": []}
        
        return {
            "x_hat": x.value.tolist(),
            "w_hat": w.value.tolist(),
            "v_hat": v.value.tolist(),
        }
