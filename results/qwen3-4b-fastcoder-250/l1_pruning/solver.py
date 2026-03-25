import numpy as np

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list]:
        v = np.array(problem["v"])
        k = problem["k"]
        n = len(v)
        
        abs_v = np.abs(v)
        indices = np.argsort(abs_v)[::-1]
        u_sorted = abs_v[indices]
        S = np.cumsum(u_sorted)
        
        j = np.arange(n)
        denominator = j + 1
        rhs = (S - k) / denominator
        conditions = u_sorted < rhs
        
        j_index = np.where(conditions)[0]
        if j_index.size == 0:
            theta = 0.0
        else:
            j_index = j_index[0]
            theta = (S[j_index] - k) / (j_index + 1)
        
        w_sorted = np.maximum(u_sorted - theta, 0)
        w = w_sorted * np.sign(v)[indices]
        
        return {"solution": w.tolist()}
