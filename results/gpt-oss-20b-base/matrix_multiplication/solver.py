import numpy as np

class Solver:
    def solve(self, problem: dict[str, list[list[float]]]) -> list[list[float]]:
        """
        Compute C = A · B for two dense matrices.
        """
        # create numpy arrays without unnecessary copies
        A = np.array(problem["A"], dtype=np.float64, copy=False)
        B = np.array(problem["B"], dtype=np.float64, copy=False)
        # perform matrix multiplication
        C = np.dot(A, B)
        # convert result back to nested Python lists
        return C.tolist()