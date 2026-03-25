# solver.py
import numpy as np
from scipy.linalg import lu

class Solver:
    def solve(self, problem, **kwargs) -> dict[str, dict[str, list[list[float]]]]:
        """
        Perform LU factorisation A = P L U via scipy's implementation.

        Parameters
        ----------
        problem : dict with key "matrix" containing a square 2‑D array.
        **kwargs : ignored.

        Returns
        -------
        dict
            {"LU": {"P": ..., "L": ..., "U": ...}}
        """
        A = problem["matrix"]

        # Ensure the input is a NumPy array (fast conversion)
        if not isinstance(A, np.ndarray):
            A = np.array(A, dtype=float)
        else:
            A = A.astype(float, copy=False)

        P, L, U = lu(A)

        # Convert matrices to nested lists for the required output format
        return {
            "LU": {
                "P": P.tolist(),
                "L": L.tolist(),
                "U": U.tolist()
            }
        }
