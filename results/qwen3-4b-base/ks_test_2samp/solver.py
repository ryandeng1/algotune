from typing import Any
import numpy as np
import scipy.stats as stats

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        sample1 = problem["sample1"]
        sample2 = problem["sample2"]
        
        if not isinstance(sample1, np.ndarray) or not isinstance(sample2, np.ndarray):
            sample1 = np.array(sample1)
            sample2 = np.array(sample2)
        
        result = stats.ks_2samp(sample1, sample2, method="auto")
        return {"statistic": result.statistic, "pvalue": result.pvalue}