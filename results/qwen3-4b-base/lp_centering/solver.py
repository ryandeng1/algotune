import numpy as np

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list]:
        c = np.array(problem["c"])
        A = np.array(problem["A"])
        b = np.array(problem["b"])
        x = np.linalg.pinv(A) @ b
        return {"solution": x.tolist()}
