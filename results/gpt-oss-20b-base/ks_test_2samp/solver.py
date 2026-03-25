import numpy as np
from scipy import stats

class Solver:
    def solve(self, problem, **kwargs) -> dict[str, Any]:
        """Compute two-sample Kolmogorov–Smirnov statistic and p‑value."""
        a = np.asarray(problem["sample1"], dtype=np.float64)
        b = np.asarray(problem["sample2"], dtype=np.float64)

        # Sort the samples
        a_sort = np.sort(a)
        b_sort = np.sort(b)

        # Compute empirical CDFs on combined sorted set
        n = a.size
        m = b.size
        idx_a = np.searchsorted(a_sort, b_sort, side="right")
        idx_b = np.searchsorted(b_sort, a_sort, side="right")

        # Differences at all sample points
        cdf_a = idx_a / n
        cdf_b = idx_b / m

        # KS statistic
        ks_stat = np.max(np.abs(cdf_a - (np.arange(1, m + 1) / m)))

        # Fairer to compute difference at unified sorted list
        all_vals = np.merge(a_sort, b_sort)
        # Actually use stats.ks_2samp for tail probability (fast enough)
        ks_res = stats.mstats.ks_2samp(a_sort, b_sort, method="asymptotic")  # uses asymptotic pvalue

        return {"statistic": ks_res.statistic, "pvalue": ks_res.pvalue}
