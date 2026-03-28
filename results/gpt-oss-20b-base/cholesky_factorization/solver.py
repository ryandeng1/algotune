import numpy as np
from typing import Any, Dict, List

class Solver:
    """
    Cholesky factorization solver.

    Given a symmetric positive‑definite matrix **A**, returns
    a nested list representation of the lower‑triangular factor **L**
    such that **A = L @ L.T**.

    The implementation relies on the highly optimised NumPy
    routine ``numpy.linalg.cholesky`` which internally uses LAPACK.
    Converting the resulting NumPy array to a plain Python list incurs
    negligible overhead for the typical problem sizes encountered
    in the benchmark suite.
    """

    def solve(self, problem: Dict[str, np.ndarray]) -> Dict[str, Dict[str, List[List[float]]]]:
        A = problem["matrix"]
        # LAPACK‑based Cholesky factorisation
        L = np.linalg.cholesky(A)
        # Convert the array to a list of lists once for the minimal overhead
        return {"Cholesky": {"L": L.tolist()}}