import scipy.stats as stats

class Solver:
    def solve(self, problem):
        """Perform two-sample KS test to obtain statistic and p-value."""
        result = stats.ks_2samp(
            problem['sample1'],
            problem['sample2'],
            method='auto'
        )
        return {'statistic': result.statistic, 'pvalue': result.pvalue}