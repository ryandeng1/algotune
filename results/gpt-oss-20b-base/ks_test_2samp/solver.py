# solver.py

import numpy as np
from scipy.stats import ks_2samp
from typing import Any, Dict


class Solver:
    """
    Optimised KS 2-sample test solver.

    The implementation converts the input samples to NumPy arrays only once
    and uses the faster asymptotic version of the Kolmogorov–Smirnov test.
    """

    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        # Convert to NumPy arrays for vectorised handling
        sample1 = np.asarray(problem["sample1"])
        sample2 = np.asarray(problem["sample2"])

        # Perform the asymptotic KS test (default is fast for large samples)
        statistic, pvalue = ks_2samp(sample1, sample2, method="asymp")

        return {"statistic": statistic, "pvalue": pvalue}