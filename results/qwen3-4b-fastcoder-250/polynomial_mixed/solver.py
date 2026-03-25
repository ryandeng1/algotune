import numpy as np

class Solver:
    def solve(self, problem: list[float]) -> list[complex]:
        roots = np.roots(problem)
        sorted_roots = np.sort(roots)[::-1]
        return sorted_roots.tolist()
