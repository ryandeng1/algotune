import numpy as np
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """Perform two-sample Kolmogorov–Smirnov test using NumPy to avoid SciPy overhead."""
        x = np.asarray(problem["sample1"])
        y = np.asarray(problem["sample2"])

        # Sort the samples
        xs = np.sort(x)
        ys = np.sort(y)

        # Cumulative counts
        n, m = xs.size, ys.size
        i = j = 0
        cdf_diff = 0.0

        # Merge step to find maximum difference between empirical CDFs
        while i < n or j < m:
            val = xs[i] if i < n and (j == m or xs[i] <= ys[j]) else ys[j]
            incr_x = 0
            incr_y = 0
            while i < n and xs[i] == val:
                incr_x += 1
                i += 1
            while j < m and ys[j] == val:
                incr_y += 1
                j += 1
            cdf_diff = max(
                cdf_diff,
                abs((i / n) - (j / m)),
                abs((i - incr_x) / n - (j - incr_y) / m)
            )

        # KS statistic
        statistic = cdf_diff

        # Approximate p-value using asymptotic formula
        # Effective sample size
        neff = n * m / (n + m)
        z = statistic * (np.sqrt(neff) + 0.12 + 0.11 / np.sqrt(neff))
        pvalue = 2 * np.exp(-2 * z * z)
        pvalue = min(max(pvalue, 0.0), 1.0)  # clip to [0,1]

        return {"statistic": statistic, "pvalue": pvalue}