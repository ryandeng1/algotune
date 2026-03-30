import numpy as np
from typing import Any, Dict, List

class Solver:
    """
    Fast SVD solver that simply forwards to numpy's highly optimised
    linalg.svd implementation.  No extra work is performed other than
    making sure the input is a NumPy array and reshaping the result
    for minimal overhead.
    """

    __slots__ = ("_np",)

    def __init__(self) -> None:
        # Using a binding prevents a repeated attribute lookup.
        self._np = np

    def solve(self, problem: Dict[str, Any]) -> Dict[str, List]:
        A = self._np.asarray(problem["matrix"], dtype=float, order="C")
        # numpy.linalg.svd is in C and is already the fastest way to
        # compute the SVD of a dense matrix.  The flags below keep the
        # call minimal.
        U, s, Vh = self._np.linalg.svd(A, compute_uv=True, full_matrices=False)
        V = Vh.T
        # Convert the result to lists only once, using view() to avoid copies
        # for large matrices.
        return {
            "U": U.view(list),
            "S": s.tolist(),
            "V": V.view(list),
        }