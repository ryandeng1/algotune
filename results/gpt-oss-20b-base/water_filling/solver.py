import numpy as np

def solve(problem: dict):
    alpha = np.asarray(problem['alpha'], dtype=float)
    P_total = float(problem.get('P_total', 0.0))
    n = alpha.size

    # Basic validation
    if n == 0 or P_total <= 0 or not np.all(alpha > 0):
        return {'x': [float('nan')] * n, 'Capacity': float('nan')}

    # Binary search for lambda such that sum(max(0, 1/lambda - alpha)) = P_total
    # lambdas > 0
    lo, hi = 1e-12, max(1.0 / (alpha + 1e-12))  # upper bound when P_total=0
    # Ensure hi is large enough
    while True:
        mid = (lo + hi) / 2.0
        x = np.where(alpha < 1.0 / mid, 1.0 / mid - alpha, 0.0)
        s = x.sum()
        if s > P_total:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-14:
            break
    lam = (lo + hi) / 2.0
    x = np.where(alpha < 1.0 / lam, 1.0 / lam - alpha, 0.0)
    # Normalize to P_total in case of numerical drift
    s = x.sum()
    if s > 1e-12:
        x = x * (P_total / s)

    capacity = np.sum(np.log(alpha + x))
    return {'x': x.tolist(), 'Capacity': float(capacity)}