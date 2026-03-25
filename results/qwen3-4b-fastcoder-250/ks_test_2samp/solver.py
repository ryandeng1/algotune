import numpy as np
import numba
from typing import Any
import scipy.stats as stats

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        sample1 = np.array(problem["sample1"], dtype=np.float64)
        sample2 = np.array(problem["sample2"], dtype=np.float64)
        n = len(sample1)
        
        @numba.jit(nopython=True)
        def compute_ks_statistic(x, y):
            x_sorted = np.sort(x)
            y_sorted = np.sort(y)
            combined = np.concatenate((x_sorted, y_sorted))
            combined_sorted = np.sort(combined)
            
            idx_x = np.searchsorted(x_sorted, combined_sorted)
            idx_y = np.searchsorted(y_sorted, combined_sorted)
            
            diff = (idx_x - idx_y) / n
            return np.max(np.abs(diff))
        
        ks_stat = compute_ks_statistic(sample1, sample2)
        
        result = stats.ks_2samp(sample1, sample2, method="auto")
        return {"statistic": ks_stat, "pvalue": result.pvalue}
