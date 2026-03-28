#!/usr/bin/env python3
from typing import Any, Dict, List
import numpy as np

def solve(problem: Dict[str, Any]) -> Dict[str, Any]:
    """Fast analytical solution to the log‑sum maximisation problem."""
    # Extract data
    alpha = np.asarray(problem["alpha"], dtype=np.float64)
    P_total = float(problem["P_total"])
    n = alpha.size

    # Input checks
    if n == 0 or P_total <= 0 or not np.all(alpha > 0):
        return {"x": [float("nan")] * n, "Capacity": float("nan")}

    # Sort alphas ascending; keep indices to restore order later
    order = np.argsort(alpha)
    alpha_sorted = alpha[order]

    # Pre‑compute cumulative sums of alphas
    cum_alpha = np.cumsum(alpha_sorted, dtype=np.float64)
    # We need to find lambda such that sum max(0, 1/lambda - alpha_i) = P_total
    # For a given λ, define m = number of i with alpha_i < 1/λ
    # Then sum = m/λ - cum_alpha[m-1]
    # Since the function is decreasing in λ we can binary‑search λ on (0,∞)

    # Helper: compute residual for given λ
    def residual(lam: float) -> float:
        if lam == 0:
            return float("inf")
        inv_lam = 1.0 / lam
        # Find the last index where alpha < inv_lam
        m = np.searchsorted(alpha_sorted, inv_lam, side="right")
        if m == 0:
            return -P_total  # all terms negative, sum 0
        return m * inv_lam - cum_alpha[m - 1] - P_total

    # Bisection bounds
    lo, hi = 1e-12, 1e12  # λ bounds
    # Adjust hi to ensure residual(hi) < 0
    while residual(hi) > 0:
        hi *= 2
    # Binary search
    for _ in range(80):  # 80 iterations give >1e-24 accuracy
        mid = (lo + hi) / 2
        if residual(mid) > 0:
            lo = mid
        else:
            hi = mid
    lam_star = (lo + hi) / 2

    inv_lam = 1.0 / lam_star
    x_sorted = np.maximum(inv_lam - alpha_sorted, 0.0)

    # Restore original order
    x = np.empty_like(x_sorted)
    x[order] = x_sorted

    # Numerical corrections (ensure sum constraint)
    cur_sum = x.sum()
    if cur_sum > 1e-9:
        x *= P_total / cur_sum

    # Capacity
    capacity = np.sum(np.log(alpha + x))

    return {"x": x.tolist(), "Capacity": float(capacity)}