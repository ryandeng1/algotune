import numpy as np
from numpy.typing import NDArray
from typing import Any

class Solver:
    def solve(self, problem: dict) -> NDArray:
        """
        Compute the cumulative integral of the 1D array using Simpson's rule in a
        fully vectorised, self‑contained implementation.  It avoids the overhead
        of calling the scipy function and works with any 1‑D numeric array
        (int, float, complex, etc.).
        """
        y: NDArray = np.asarray(problem["y"])
        dx: float = float(problem.get("dx", 1.0))

        n = y.size
        if n < 2:
            return np.full_like(y, np.nan)

        # Build the weight pattern for Simpson: 1 4 2 4 2 ... 4 1
        # For odd n the last weight is 1, for even n the last weight is 4.
        # Duplicate the pattern to match the length, trimming to exact size.
        base = np.array([1, 4, 2], dtype=y.dtype)
        full = []
        i = 0
        while len(full) < n:
            if i == 0:
                full.append(base[0])
            elif i == n - 1:
                full.append(base[0]) if (n % 2 == 1) else full.append(base[1])
                break
            else:
                full.append(base[1 if (i % 2) else 2])
            i += 1
        weights = np.array(full, dtype=y.dtype)

        # Weighted cumulative sum and scaling
        cum = np.cumsum(y * weights) * (dx / 3.0)

        return cum