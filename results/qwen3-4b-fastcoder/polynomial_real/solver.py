from typing import Any
import numpy as np

class Solver:
    def solve(self, problem: list[float]) -> list[float]:
        roots = np.roots(problem)
        roots = np.real_if_close(roots, tol=1e-3)
        roots = np.sort(roots)[::-1]
        return roots.tolist()