from scipy.stats import ks_2samp

class Solver:
    def solve(self, problem: dict[str, object]) -> dict[str, float]:
        """Perform two-sample KS test to get statistic and pvalue."""
        result = ks_2samp(problem['sample1'], problem['sample2'], method='auto')
        return {'statistic': result.statistic, 'pvalue': result.pvalue}