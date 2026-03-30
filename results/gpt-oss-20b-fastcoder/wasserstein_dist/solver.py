# solver.py

import numpy as np
from scipy.stats import wasserstein_distance


class Solver:
    """Compute 1‑D Wasserstein distance between two discrete distributions.

    The distributions are defined by their values `u` and `v` over the
    discrete support ``[1, 2, …, n]`` where ``n`` is the length of the
    value lists.  The implementation relies on :func:`scipy.stats.wasserstein_distance`
    but avoids the overhead of building Python ``list`` objects for the support
    and keeps the control flow free of ``try``/``except`` blocks.
    """

    def solve(self, problem: dict[str, list[float]]) -> float:
        """
        Parameters
        ----------
        problem: dict with keys 'u' and 'v' each mapping to a list of floats.

        Returns
        -------
        float
            Wasserstein distance between the two distributions.
        """
        u = np.asarray(problem["u"], dtype=np.float64, order="C")
        v = np.asarray(problem["v"], dtype=np.float64, order="C")
        n = u.size

        # Guard against empty inputs that would raise an exception in
        # scipy.stats.wasserstein_distance.
        if n == 0:
            return 0.0

        # Support is the same for both arrays: 1, 2, ..., n
        support = np.arange(1, n + 1, dtype=np.int64)

        return wasserstein_distance(support, support, u, v)