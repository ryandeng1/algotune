from typing import Any, Dict, List
import numpy as np


class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, List]:
        """
        Compute the singular value decomposition of the input matrix.

        Parameters
        ----------
        problem : dict
            Must contain the key "matrix" mapping to a numpy array of shape (n, m).

        Returns
        -------
        dict
            Keys:
                "U": left singular vectors (n, k) array,
                "S": singular values (k,) array,
                "V": right singular vectors (k, m) array.
        """
        A = problem["matrix"]
        # Compute compact SVD. Use the default algorithm which is fast for most shapes.
        U, S, Vh = np.linalg.svd(A, full_matrices=False)
        V = Vh.T  # Convert V^H to V

        # Return as plain numpy arrays; caller can convert to list if needed.
        return {"U": U, "S": S, "V": V}