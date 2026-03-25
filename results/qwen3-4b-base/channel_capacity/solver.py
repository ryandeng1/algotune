import math
import numpy as np
import cvxpy as cp
from scipy.special import xlogy

class Solver:
    def solve(self, problem: dict) -> dict:
        P = np.array(problem["P"])
        m, n = P.shape
        if not (P.shape == (n, m)):
            return None
        if not (n > 0 and m > 0):
            return None
        if not np.allclose(np.sum(P, axis=0), 1.0, atol=1e-6):
            return None
        
        x = cp.Variable(n)
        y = P @ x
        c = np.sum(xlogy(P, P), axis=0) / math.log(2)
        mutual_information = c @ x + cp.sum(cp.entr(y) / math.log(2))
        objective = cp.Maximize(mutual_information)
        constraints = [cp.sum(x) == 1, x >= 0]
        
        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=cp.ECOS, abstol=1e-9, reltol=1e-9)
        except cp.SolverError:
            return None
        except Exception:
            return None
        
        if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            return None
        if prob.value is None:
            return None
        
        return {"x": x.value.tolist(), "C": prob.value}
