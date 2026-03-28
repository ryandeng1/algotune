from typing import Any
import numpy as np

class Solver:
    def solve(self, problem: dict[str, list[list[float]]]) -> list[list[float]]:
        A = np.array(problem["A"])
        B = np.array(problem["B"])
        return (A @ B).tolist()