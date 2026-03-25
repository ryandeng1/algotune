import numpy as np

class Solver:
    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, dict[str, list[list[float]]]]:
        A = problem["matrix"]
        Q, R = np.linalg.qr(A, mode="reduced")
        return {"QR": {"Q": Q.tolist(), "R": R.tolist()}}
