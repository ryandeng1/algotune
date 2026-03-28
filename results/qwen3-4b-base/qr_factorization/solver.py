from typing import Any
import numpy as np


class Solver:
    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, dict[str, list[list[float]]]]:
        matrix = problem["matrix"]
        Q, R = np.linalg.qr(matrix, mode="reduced")
        return {"QR": {"Q": Q.tolist(), "R": R.tolist()}}