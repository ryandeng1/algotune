import numpy as np
import scipy.linalg
import numba

@numba.jit(nopython=True)
def to_list(X):
    result = []
    for i in range(X.shape[0]):
        row = []
        for j in range(X.shape[1]):
            row.append(X[i, j])
        result.append(row)
    return result

class Solver:
    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, dict[str, list[list[complex]]]]:
        A = problem["matrix"]
        try:
            X = scipy.linalg.sqrtm(A, disp=False)
            return {"sqrtm": {"X": to_list(X)}}
        except Exception:
            return {"sqrtm": {"X": []}}
