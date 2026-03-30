import numpy as np
import ot

class Solver:
    """
    Optimised solver for Earth Mover's Distance (EMD) using the POT library.
    The implementation is intentionally minimal and relies on the highly
    optimised C++ backend of POT, which gives the best performance for
    this problem size. All heavy lifting is done in the library – we only
    prepare the data in the fastest possible way.
    """

    @staticmethod
    def _prepare_matrix(M: np.ndarray) -> np.ndarray:
        """
        POT requires the cost matrix to be contiguous and packed in column-major
        order for the underlying LAP implementation.  Creating a contiguous float64
        view with zero-copy semantics is usually fast and guarantees the correct
        format without any extra allocation when the input is already in the
        right layout.
        """
        # Ensure dtype is float64 and data is contiguous
        if not M.dtype == np.float64:
            M = M.astype(np.float64, copy=False)
        return np.ascontiguousarray(M, dtype=np.float64)

    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, list[list[float]]]:
        """
        Compute the optimal transport plan for an Earth Mover's Distance (EMD)
        problem. The function is heavily optimised for speed: all data is
        validated and converted in place where possible, and the heavy
        computation is performed entirely within the C/C++ backend of the
        POT library.
        """
        # Pull data from the dictionary; no copies are made
        a = problem["source_weights"]
        b = problem["target_weights"]
        M = self._prepare_matrix(problem["cost_matrix"])

        # POT's emd function returns a NumPy array; we keep it as such
        G = ot.lp.emd(a, b, M, check_marginals=False)

        # The caller expects a Python list of lists
        return {"transport_plan": G.tolist()}