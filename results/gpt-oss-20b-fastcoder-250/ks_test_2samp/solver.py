# solver.py
import numpy as np
from scipy.special import gamma
from scipy.stats._continuous_distns import kstwobign
from typing import Any, Dict

class Solver:
    def _ks_statistic(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute two‑sample KS statistic for arrays a and b."""
        # Sort the input arrays
        a_sorted = np.sort(a)
        b_sorted = np.sort(b)
        na = len(a_sorted)
        nb = len(b_sorted)

        # Merge the two sorted arrays while keeping track of indices
        i = j = 0
        cdf_a = cdf_b = 0.0
        max_diff = 0.0

        while i < na or j < nb:
            if j == nb or (i < na and a_sorted[i] <= b_sorted[j]):
                val = a_sorted[i]
                cdf_a += 1 / na
                i += 1
            else:
                val = b_sorted[j]
                cdf_b += 1 / nb
                j += 1

            diff = abs(cdf_a - cdf_b)
            if diff > max_diff:
                max_diff = diff

        return max_diff

    def _ks_pvalue(self, d: float, n1: int, n2: int) -> float:
        """Approximate two‑sample KS p‑value using the Kolmogorov distribution."""
        # Effective sample size for the asymptotic formula
        n_eff = (n1 * n2) / (n1 + n2)
        # Scaling factor
        lambda_ = d * np.sqrt(n_eff)
        # Use the complementary cumulative distribution of KSB
        # kstwobign.cdf gives CDF, so subtract from 1
        return 1.0 - kstwobign.cdf(lambda_)

    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Compute two‑sample Kolmogorov–Smirnov statistic and p‑value."""
        a = np.asarray(problem["sample1"], dtype=np.float64)
        b = np.asarray(problem["sample2"], dtype=np.float64)

        stat = self._ks_statistic(a, b)
        pval = self._ks_pvalue(stat, len(a), len(b))

        # Ensure numerical stability
        if np.isnan(stat):
            stat = 0.0
        if np.isnan(pval):
            pval = 1.0

        return {"statistic": float(stat), "pvalue": float(pval)}
