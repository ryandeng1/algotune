import numpy as np

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list]:
        v = np.array(problem["v"])
        k = problem["k"]
        
        abs_v = np.abs(v)
        indices = np.argpartition(abs_v, -k)
        top_k_indices = indices[-k:]
        
        pruned = np.zeros_like(v)
        pruned[top_k_indices] = v[top_k_indices]
        
        return {"solution": pruned.tolist()}
