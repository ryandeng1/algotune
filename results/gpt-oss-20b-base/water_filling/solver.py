import numpy as np
from typing import Any, Dict


def solve(problem: Dict[str, Any]) -> Dict[str, Any]:
    """
    Maximize Σ log(alpha_i + x_i) s.t. Σ x_i = P_total, x_i ≥ 0.
    Optimal solution can be found by a water‑filling algorithm
    in O(n log n).
    """
    alpha = np.asarray(problem.get("alpha", []), dtype=float)
    P_total = float(problem.get("P_total", 0.0))
    n = alpha.size

    # Basic input validation
    if n == 0 or P_total <= 0 or not np.all(alpha > 0):
        return {"x": [float("nan")] * n, "Capacity": float("nan")}

    # Sort alphas ascendingly and keep original indices
    idx = np.argsort(alpha)
    alpha_sorted = alpha[idx]
    # Prefix sums of alphas for fast sum over suffixes
    pref_alpha = np.concatenate([[0.0], np.cumsum(alpha_sorted)])

    # Find the largest k such that 1/λ > alpha_sorted[k]
    # λ = n / (P_total + Σ alpha)
    # For a given subset S (size k+1) we use λ = k+1 / (P_total + Σ_{i∈S} alpha_i)
    k = -1
    for i in range(n):
        k = i
        lam = (k + 1) / (P_total + pref_alpha[k + 1])
        if lam <= alpha_sorted[k]:
            k -= 1
            break

    if k == -1:
        # No positive allocation (unlikely because alpha>0 and P_total>0)
        x = np.zeros(n)
    else:
        lam = (k + 1) / (P_total + pref_alpha[k + 1])
        x_sorted = np.maximum(1 / lam - alpha_sorted, 0.0)
        # Explicitly enforce sum==P_total
        x_sorted = x_sorted / np.sum(x_sorted) * P_total if np.sum(x_sorted) > 0 else x_sorted
        # Map back to original order
        x = np.empty(n)
        x[idx] = x_sorted

    # Compute capacity
    terms = np.log(alpha + x)
    if not np.all(np.isfinite(terms)):
        capacity = float("nan")
    else:
        capacity = float(np.sum(terms))

    return {"x": x.tolist(), "Capacity": capacity}