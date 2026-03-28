import numpy as np

class Solver:
    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, dict[str, list[list[float]]]]:
        # Compute the Cholesky decomposition using NumPy's highly optimised routine.
        return {"Cholesky": {"L": np.linalg.cholesky(problem["matrix"])}}