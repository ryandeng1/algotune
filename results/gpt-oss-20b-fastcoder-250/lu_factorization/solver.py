import numpy as np
from scipy.linalg import lu

class Solver:
    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, dict[str, list[list[float]]]]:
        """
        Compute the LU factorization A = P @ L @ U of a square matrix A.

        Parameters
        ----------
        problem : dict
            Dictionary containing the key "matrix" with a 2-D square numpy array.

        Returns
        -------
        dict
            Dictionary of the form
            {
                "LU": {
                    "P": [[ ... ], ...],
                    "L": [[ ... ], ...],
                    "U": [[ ... ], ...]
                }
            }
        """
        A = problem["matrix"]
        P, L, U = lu(A)
        return {"LU": {"P": P.tolist(), "L": L.tolist(), "U": U.tolist()}}
