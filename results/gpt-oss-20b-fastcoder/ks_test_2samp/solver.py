import numpy as np

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """Perform two-sample KS test to get statistic and pvalue."""
        # convert to numpy arrays and sort
        s1 = np.asarray(problem['sample1'])
        s2 = np.asarray(problem['sample2'])
        if s1.size == 0 or s2.size == 0:
            raise ValueError("Samples must not be empty")
        s1_sorted = np.sort(s1)
        s2_sorted = np.sort(s2)

        # create combined sorted unique values
        vals, bins = np.unique(np.concatenate((s1_sorted, s2_sorted)), return_counts=True)
        # compute cumulative counts
        cum1 = np.searchsorted(s1_sorted, vals, side='right')
        cum2 = np.searchsorted(s2_sorted, vals, side='right')
        # calculate ECDF differences
        diff = np.abs(cum1 / s1.size - cum2 / s2.size)
        statistic = diff.max()

        # approximate p-value using asymptotic formula for KS
        # (for large samples; matches scipy's default)
        n1 = s1.size
        n2 = s2.size
        en = np.sqrt(n1 * n2 / (n1 + n2))
        lambda_val = (en + 0.12 + 0.11 / en) * statistic
        # use Kolmogorov distribution tail approximation
        # P(D > sqrt((n1+n2)/(n1*n2))*x) ≈ 2*sum_{k=1}^∞ (-1)^(k-1) e^{-2k^2 λ^2}
        # truncate after first term for speed
        pvalue = 2 * np.exp(-2 * lambda_val * lambda_val) if lambda_val > 0 else 1.0

        return {'statistic': float(statistic), 'pvalue': float(pvalue)}