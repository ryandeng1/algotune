from typing import List, Tuple
import numpy as np

class Solver:
    """
    Fast 1‑D correlation solver.

    The implementation uses numpy's vectorised `correlate` routine which is heavily
    optimised in C.  There is no branching inside the loop and the algorithm
    is completely memory‑local, so it scales linearly with the total input size.
    """

    def __init__(self):
        self.mode: str = "full"

    def solve(self, problem: List[Tuple[np.ndarray, np.ndarray]]) -> List[np.ndarray]:
        """
        Compute the 1D correlation for each valid pair in the problem list.

        Parameters
        ----------
        problem : list
            A list of (a, b) tuples, where ``a`` and ``b`` are 1‑D numpy arrays.

        Returns
        -------
        list
            A list of correlation results (numpy arrays).  Pairs which are
            discarded by the "valid" mode are omitted from the output.
        """
        mode = self.mode
        results: List[np.ndarray] = []

        # Use a local reference to the function to avoid global lookups
        _correlate = np.correlate

        # The 'valid' mode discards pairs where len(b) > len(a)
        if mode == "valid":
            for a, b in problem:
                if b.size > a.size:
                    continue
                results.append(_correlate(a, b, mode=mode))
        else:
            # For `full`, `same`, or any other mode there is no filtering
            for a, b in problem:
                results.append(_correlate(a, b, mode=mode))

        return results