import numpy as np

class Solver:
    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, dict[str, list[list[float]]]]:
        A = np.asarray(problem.get('A'))
        B = np.asarray(problem.get('B'))
        if A.ndim != 2 or B.ndim != 2 or A.shape != B.shape:
            return {}
        M = B @ A.T
        U, _ , Vt = np.linalg.svd(M, full_matrices=False)
        G = U @ Vt
        return {'solution': G.tolist()}