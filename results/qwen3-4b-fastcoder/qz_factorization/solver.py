from typing import Any
import numpy as np
from scipy.linalg import qz


class Solver:
    def solve(
        self, problem: dict[str, list[list[float]]]
    ) -> dict[str, dict[str, list[list[float | complex]]]]:
        A = np.array(problem["A"])
        B = np.array(problem["B"])
        AA, BB, Q, Z = qz(A, B, output="real")
        return {"QZ": {"AA": AA.tolist(), "BB": BB.tolist(), "Q": Q.tolist(), "Z": Z.tolist()}}