import numpy as np
from scipy.stats import norm

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        sample1 = np.array(problem['sample1'])
        sample2 = np.array(problem['sample2'])
        n = len(sample1)
        
        sample1_sorted = np.sort(sample1)
        sample2_sorted = np.sort(sample2)
        
        indices1 = np.searchsorted(sample1_sorted, sample2_sorted)
        ecdf1 = indices1.astype(float) / n
        
        indices2 = np.searchsorted(sample2_sorted, sample1_sorted)
        ecdf2 = indices2.astype(float) / n
        
        statistic = np.max(np.abs(ecdf1 - ecdf2))
        
        p_value = 2 * (1 - norm.cdf(np.sqrt(n) * statistic))
        
        return {"statistic": statistic, "pvalue": p_value}
