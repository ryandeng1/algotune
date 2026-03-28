from typing import Any
import numpy as np


class Solver:
    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, dict[str, list[list[float]]]]:
        A = problem["matrix"]
        L = np.linalg.cholesky(A)
        solution = {"Cholesky": {"L": L.tolist()}}
        return solution