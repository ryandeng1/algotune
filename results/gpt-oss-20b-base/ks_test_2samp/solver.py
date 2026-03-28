from typing import Any, Dict
import numpy as np

class Solver:
    # Pre‑allocate temporary arrays once for reuse
    _tmp_union: np.ndarray | None = None
    _tmp_cnt1: np.ndarray | None = None
    _tmp_cnt2: np.ndarray | None = None

    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Perform two sample KS test (fast NumPy implementation)."""
        a = np.asarray(problem['sample1'])
        b = np.asarray(problem['sample2'])
        n1, n2 = a.size, b.size

        # sort inputs
        a_sort = np.sort(a)
        b_sort = np.sort(b)

        # build union of unique points
        if (self._tmp_union is None or
            self._tmp_union.size < a_sort.size + b_sort.size):
            self._tmp_union = np.empty(a_sort.size + b_sort.size, dtype=a_sort.dtype)
        union = np.sort(np.concatenate((a_sort, b_sort)))
        # unique with original indices
        uniq, idx = np.unique(union, return_index=True)

        # counts of each point in each sample
        if (self._tmp_cnt1 is None or self._tmp_cnt1.size < uniq.size):
            self._tmp_cnt1 = np.empty(uniq.size, dtype=int)
        if (self._tmp_cnt2 is None or self._tmp_cnt2.size < uniq.size):
            self._tmp_cnt2 = np.empty(uniq.size, dtype=int)

        # number of occurrences in a and b at each unique value
        self._tmp_cnt1[:] = np.diff(np.concatenate(([0], np.searchsorted(a_sort, uniq[ind:], side='right'))))
        self._tmp_cnt2[:] = np.diff(np.concatenate(([0], np.searchsorted(b_sort, uniq[ind:], side='right'))))

        # cumulative distribution functions
        cdf1 = np.cumsum(self._tmp_cnt1) / n1
        cdf2 = np.cumsum(self._tmp_cnt2) / n2

        # KS statistic
        statistic = np.max(np.abs(cdf1 - cdf2))
        # approximate two‑sided p‑value (use same as scipy)
        # The exact p-value requires more work; here we use asymptotic chi‑square approximation
        en = np.sqrt(n1 * n2 / (n1 + n2))
        z = (statistic - 1/(2*en)) * np.sqrt(en)
        pvalue = 2 * (1 - stats_norm_cdf(z))
        return {'statistic': float(statistic), 'pvalue': float(pvalue)}

# Auxiliary: fast normal CDF using np.erf
def stats_norm_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1 + np.erf(x / np.sqrt(2)))