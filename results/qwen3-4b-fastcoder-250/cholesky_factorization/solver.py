import numpy as np

class Solver:
    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, dict[str, list[list[float]]]]:
        A = problem["matrix"]
        L = np.linalg.cholesky(A)
        return {"Cholesky": {"L": L.tolist()}}
