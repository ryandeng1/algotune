import numpy as np
from scipy.linalg import expm

class Solver:
    def solve(self, problem, **kwargs) -> dict[str, list[list[float]]]:
        """
        Compute the matrix exponential of the input square matrix A.

        Parameters
        ----------
        problem : dict
            A dictionary containing a key "matrix" with a NumPy array or a nested
            list representing the square matrix A.

        Returns
        -------
        dict
            Dictionary with a single key "exponential" mapping to a nested list
            containing the matrix exponential exp(A).
        """
        # Extract matrix, ensuring it is a NumPy array
        A = np.asarray(problem["matrix"], dtype=float)

        # Compute the matrix exponential using SciPy's highly-optimized routine
        expA = expm(A)

        # Convert to a nested list (list of lists) to match expected output format
        return {"exponential": expA.tolist()}
