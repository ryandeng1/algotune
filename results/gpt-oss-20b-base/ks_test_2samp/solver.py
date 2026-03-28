from __future__ import annotations
from typing import Any, Dict

class Solver:
    """Fast two‑sample Kolmogorov–Smirnov test implementation."""
    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        a, b = problem["sample1"], problem["sample2"]
        n1, n2 = len(a), len(b)
        if n1 == 0 or n2 == 0:
            raise ValueError("Both samples must contain at least one value.")

        # Sort the samples
        a_sorted = sorted(a)
        b_sorted = sorted(b)

        # Two‑pointer scan to compute maximum absolute difference of empirical CDFs
        i = j = 0
        cdf_a = cdf_b = 0.0
        max_diff = 0.0

        while i < n1 or j < n2:
            val_a = a_sorted[i] if i < n1 else float("inf")
            val_b = b_sorted[j] if j < n2 else float("inf")

            if val_a <= val_b:
                i += 1
                cdf_a = i / n1
            else:
                j += 1
                cdf_b = j / n2

            diff = abs(cdf_a - cdf_b)
            if diff > max_diff:
                max_diff = diff

        D = max_diff

        # Asymptotic p‑value calculation using the Kolmogorov distribution
        # p ≈ 2 * Σ (-1)^(k-1) exp(-2 k² D² n1 n2/(n1+n2))
        # Summation stops when additional terms are < 1e-12
        n = n1 + n2
        lambda_ = D * (n ** 0.5)
        if lambda_ == 0:
            p = 1.0
        else:
            term = -2 * lambda_ ** 2
            k = 1
            p_sum = 0.0
            while True:
                p_term = (2 if k & 1 else -2) * (-1) ** (k - 1) * __import__("math").exp(term * k * k)
                p_sum += p_term
                if abs(p_term) < 1e-12:
                    break
                k += 1
            p = min(max(p_sum, 0.0), 1.0)

        return {"statistic": D, "pvalue": p}