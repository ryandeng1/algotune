import numpy as np

class Solver:
    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, dict[str, list[list[float]]]]:
        A = problem.get("A")
        B = problem.get("B")
        if A is None or B is None:
            return {}
        A = np.array(A)
        B = np.array(B)
        if A.shape != B.shape:
            return {}
        M = B @ A.T
        U, _, Vt = np.linalg.svd(M)
        G = U @ Vt
        return {"solution": G.tolist()}
