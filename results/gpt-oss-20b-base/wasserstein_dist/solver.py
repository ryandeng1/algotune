# solver.py
import numpy as np
from scipy.stats import wasserstein_distance

class Solver:
    """
    A lightweight solver that computes the 1‑dimensional Wasserstein distance
    using SciPy's fast C implementation.

    The positional grid is fixed at indices 1 … N where N is the number of mass
    points supplied in each problem.  The grid is pre‑computed for every
    problem length that occurs.  For repeated calls with the same size the
    same grid numpy array is reused, avoiding duplicate list → array conversion.
    """

    # Cache the grid for each length that has been seen
    _grid_cache: dict[int, np.ndarray] = {}

    def __init__(self) -> None:
        # Nothing to do on initialization that will impact runtime of `solve`.
        pass

    def _get_grid(self, n: int) -> np.ndarray:
        """Return a 1‑based grid 1 … n as a NumPy array."""
        if n not in self._grid_cache:
            self._grid_cache[n] = np.arange(1, n + 1, dtype=np.float64)
        return self._grid_cache[n]

    def solve(self, problem: dict[str, list[float]]) -> float:
        """
        Compute the 1‑dimensional Wasserstein (Earth‑Mover) distance between
        two discrete probability distributions `u` and `v`.

        The input `problem` must contain the keys 'u' and 'v' mapping to
        lists of floats of equal length.

        Returns the distance as a float.
        """
        u = np.asarray(problem['u'], dtype=np.float64)
        v = np.asarray(problem['v'], dtype=np.float64)

        # Defensive check: in case of mismatch or empty input
        n = u.size
        if n == 0 or v.size != n:
            return float(n)

        grid = self._get_grid(n)

        # SciPy's wasserstein_distance uses a fast C implementation
        # and expects numpy arrays for both grid and weights.
        return wasserstein_distance(grid, grid, u, v)