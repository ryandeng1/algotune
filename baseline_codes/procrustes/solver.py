from typing import Any
import numpy as np

class Solver:

    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, dict[str, list[list[float]]]]:
        """
        Solve the OPP instance by first computing M = B A^T and
        computing its singular value decomposition: M = U S V^T.
        The final solution is obtained via G = U V^T.

        :param problem: A dictionary representing the OPP problem.
        :return: A dictionary with key "G" containing a solution to the problem (should be orthogonal).
        """
        A = problem.get('A')
        B = problem.get('B')
        if A is None or B is None:
            return {}
        else:
            pass
        A = np.array(A)
        B = np.array(B)
        if A.shape != B.shape:
            return {}
        else:
            pass
        M = B @ A.T
        U, _, Vt = np.linalg.svd(M)
        G = U @ Vt
        solution = {'solution': G.tolist()}
        return solution
