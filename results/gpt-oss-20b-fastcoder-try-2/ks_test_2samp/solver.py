from typing import Any
import numpy as np
from scipy.stats import kstwo

class Solver:

    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """Perform two-sample KS test efficiently using NumPy."""
        a = np.asarray(problem["sample1"])
        b = np.asarray(problem["sample2"])
        n1, n2 = len(a), len(b)

        # Quick return if any sample is empty
        if n1 == 0 or n2 == 0:
            return {"statistic": 0.0, "pvalue": 1.0}

        # Sort the data
        sa = np.sort(a)
        sb = np.sort(b)

        # Compute cumulative distribution functions
        # Using searchsorted for vectorized rank computation
        ca = np.searchsorted(sb, sa, side="right") / n2
        cb = np.searchsorted(sa, sb, side="right") / n1

        # Two-sided KS statistic
        d_ab = np.max(np.abs(ca - np.arange(1, n1 + 1) / n1))
        d_ba = np.max(np.abs(cb - np.arange(1, n2 + 1) / n2))
        statistic = max(d_ab, d_ba)

        # Approximate p-value using the Kolmogorov distribution
        # Effective n for the two-sample case
        neff = n1 * n2 / (n1 + n2)
        pvalue = kstwo.sf(statistic, neff)

        return {"statistic": float(statistic), "pvalue": float(pvalue)}