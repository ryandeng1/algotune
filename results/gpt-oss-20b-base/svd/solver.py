import numpy as np
from typing import Any, Dict

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Compute the singular value decomposition of the given matrix A.

        Parameters
        ----------
        problem : dict
            Must contain the key "matrix" whose value is a 2‑D array‑like object.
        kwargs : any
            Additional keyword arguments are ignored.

        Returns
        -------
        dict
            A dictionary with keys "U", "S", "V" containing the left singular vectors,
            singular values and right singular vectors respectively. All values are
            returned as NumPy arrays (compatible with the reference checker).
        """
        A = np.array(problem["matrix"], dtype=float)
        U, s, Vh = np.linalg.svd(A, full_matrices=False)
        V = Vh.T
        return {"U": U, "S": s, "V": V}
