import numpy as np

class Solver:
    def solve(self, problem: list[float]) -> list[complex]:
        roots = np.roots(problem)
        return sorted(roots, key=lambda z: (z.real, z.imag), reverse=True)
