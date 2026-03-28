from typing import Any
import scipy.stats as stats

class Solver:

    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """Peform two sample ks test to get statistic and pvalue."""
        result = stats.ks_2samp(problem['sample1'], problem['sample2'], method='auto')
        return {'statistic': result.statistic, 'pvalue': result.pvalue}
