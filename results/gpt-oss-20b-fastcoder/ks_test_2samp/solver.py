from typing import Any, Dict
import numpy as np
import math

class Solver:
    def _ks_2samp(self, a: np.ndarray, b: np.ndarray):
        na = a.size
        nb = b.size
        if na == 0 or nb == 0:
            return 1.0, 0.0

        # Sort the samples
        a = np.sort(a)
        b = np.sort(b)

        i = j = 0
        max_diff = 0.0

        # Two‑pointer scan to compute the empirical CDF difference
        while i < na or j < nb:
            if j == nb or (i < na and a[i] <= b[j]):
                i += 1
                cdf_a = i / na
                cdf_b = j / nb
            else:
                j += 1
                cdf_a = i / na
                cdf_b = j / nb
            diff = abs(cdf_a - cdf_b)
            if diff > max_diff:
                max_diff = diff

        # Asymptotic p‑value approximation (valid for large samples)
        n_eff = na * nb / (na + nb)
        lambda_val = max_diff * math.sqrt(n_eff * 2.0)
        if lambda_val < 0.01:
            pvalue = 1.0
        else:
            # Complementary error function approximation:
            pvalue = 2.0 * sum(
                (-1) ** (k - 1) * math.exp(-2.0 * (k * lambda_val) ** 2) for k in range(1, 100)
            )
            pvalue = max(0.0, min(1.0, pvalue))

        return max_diff, pvalue

    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Perform two-sample KS test using a fast NumPy implementation."""
        sample1 = np.asarray(problem["sample1"])
        sample2 = np.asarray(problem["sample2"])
        statistic, pvalue = self._ks_2samp(sample1, sample2)
        return {"statistic": statistic, "pvalue": pvalue}