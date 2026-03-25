import numpy as np

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list]:
        A = np.array(problem["matrix"])
        U, s, Vh = np.linalg.svd(A, full_matrices=False)
        V = Vh.T
        return {"U": U, "S": s, "V": V}
