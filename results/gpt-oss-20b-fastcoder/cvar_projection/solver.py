from typing import Any, Dict, List
import numpy as np

def solve(problem: Dict[str, Any]) -> Dict[str, List[float]]:
    """
    Fast projection onto a CVaR constraint set.

    Parameters
    ----------
    problem : dict
        Must contain:
        * 'x0'            – Point to project (array‑like)
        * 'loss_scenarios' – Loss matrix (n_scenarios × n_dims)
        * 'beta'          – Confidence level (default 0.95)
        * 'kappa'         – Scale parameter for CVaR bound (default 1.0)

    Returns
    -------
    dict
        Dictionary with key 'x_proj' mapping to the projected point.

    Notes
    -----
    This implementation relies on a fast analytical projection.  If the
    unconstrained point already satisfies the CVaR constraint we simply
    return it.  Otherwise we solve the problem via a small linear program
    in the dual space that can be evaluated with numpy only, yielding a
    significant speed‑up over a generic solver such as CVXPY.
    """
    # Extract and cast data
    x0 = np.asarray(problem['x0'], dtype=float)
    A = np.asarray(problem['loss_scenarios'], dtype=float)
    beta = float(problem.get('beta', 0.95))
    kappa = float(problem.get('kappa', 1.0))

    n_scenarios, n_dims = A.shape
    k = int((1.0 - beta) * n_scenarios)

    # Check if x0 already satisfies the CVaR constraint
    scores = A @ x0
    if np.sum(np.partition(scores, -k)[-k:]) <= kappa * k:
        return {'x_proj': x0.tolist()}

    # Dual formulation – solve for multiplier λ that tightens the k largest scores
    # We sort the scores and binary search λ so that the sum of the k largest
    # "treated" scores equals the bound.
    def constraint_feasible(lam: float) -> bool:
        # Tilted scores: max(0, score - lam)
        tilted = np.maximum(scores - lam, 0.0)
        return np.sum(np.partition(tilted, -k)[-k:]) <= kappa * k

    # Binary search for λ in [0, max(scores)] (scores can be negative)
    low, high = 0.0, np.max(scores)
    for _ in range(30):
        mid = (low + high) / 2.0
        if constraint_feasible(mid):
            high = mid
        else:
            low = mid

    lam = high
    # Compute the gradient of the dual objective
    tilt = np.maximum(scores - lam, 0.0)
    # Selected scenarios (k largest after tilting)
    idx = np.argpartition(tilt, -k)[-k:]  # unsorted
    # Solve for projection as x = x0 - (1/2) * A^T * μ where μ is λ times indicator
    mu = np.zeros(n_scenarios)
    mu[idx] = lam

    # Compute projection: solve normal equations (2I) x = 2x0 - A^T μ
    # Equivalent to x = x0 - 0.5 * A^T μ
    x_proj = x0 - 0.5 * A.T @ mu
    return {'x_proj': x_proj.tolist()}