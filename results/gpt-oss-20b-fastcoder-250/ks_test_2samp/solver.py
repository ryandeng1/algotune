from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """Perform two‑sample Kolmogorov–Smirnov test using only NumPy."""
        import numpy as np

        a = np.asarray(problem["sample1"])
        b = np.asarray(problem["sample2"])

        na, nb = len(a), len(b)
        # Sort the samples
        sa = np.sort(a)
        sb = np.sort(b)

        # Cumulative counts
        i = j = 0
        cdf_a = 0.0
        cdf_b = 0.0
        max_diff = 0.0

        # Merge the two sorted arrays
        while i < na or j < nb:
            if j == nb or (i < na and sa[i] <= sb[j]):
                val = sa[i]
                while i < na and sa[i] == val:
                    i += 1
                cdf_a = i / na
            else:
                val = sb[j]
                while j < nb and sb[j] == val:
                    j += 1
                cdf_b = j / nb
            diff = abs(cdf_a - cdf_b)
            if diff > max_diff:
                max_diff = diff

        # Approximate p‑value using the asymptotic formula
        n = na + nb
        lambda_val = (np.sqrt(n) + 0.12 + 0.11 / np.sqrt(n)) * max_diff
        # Use complementary error function for p‐value approximation
        p_value = 2 * np.exp(-2 * lambda_val ** 2) if lambda_val > 0 else 1.0

        return {"statistic": float(max_diff), "pvalue": float(p_value)}