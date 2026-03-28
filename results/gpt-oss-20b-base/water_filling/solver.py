import numpy as np

def solve(problem):
    alpha = np.asarray(problem['alpha'], dtype=float)
    P_total = float(problem['P_total'])
    n = alpha.size

    # Quick checks
    if n == 0 or P_total <= 0 or not np.all(alpha > 0):
        return {'x': [float('nan')] * n, 'Capacity': float('nan')}

    # Water-filling style solution (KKT for maximizing sum log(alpha + x))
    # sort alphas and remember original indices
    idx = np.argsort(alpha)
    alpha_sorted = alpha[idx]
    cum_alpha = np.cumsum(alpha_sorted)

    # Find the largest k such that the corresponding lambda <= 1/alpha_k
    sol = None
    for k in range(1, n + 1):
        lambda_val = k / (P_total + cum_alpha[k - 1])
        if lambda_val <= 1.0 / alpha_sorted[k - 1]:
            x_temp = 1.0 / lambda_val - alpha_sorted
            # other entries beyond k are zero
            x_temp[k:] = 0.0
            sol = x_temp
            break

    if sol is None:
        # All entries saturated to zero if lambda too large
        sol = np.zeros(n)

    # Restore to original order
    x = np.empty_like(sol)
    x[idx] = sol

    # Ensure feasibility (numerical robustness)
    x = np.maximum(x, 0.0)
    x = x * P_total / np.maximum(np.sum(x), 1e-12)

    # Compute capacity
    if np.any(np.isnan(x)) or np.any(np.isinf(x)):
        capacity = float('nan')
    else:
        capacity = float(np.sum(np.log(alpha + x)))

    return {'x': x.tolist(), 'Capacity': capacity}