import numpy as np

class Solver:
    def solve(self, problem: dict[str, Any]) -> list[float]:
        A = np.array(problem["A"])
        b = np.array(problem["b"])
        x = np.linalg.solve(A, b)
        return x.tolist()
