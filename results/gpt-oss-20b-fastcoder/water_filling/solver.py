import numpy as np

class Solver:
    """
    Analytic solver for the problem:
        maximize sum log(alpha_i + x_i)
        subject to sum x_i = P_total, x_i >= 0
    """

    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        # Input extraction & validation
        alpha = np.asarray(problem["alpha"], dtype=np.float64, order="C")
        P_total = float(problem["P_total"])
        n = alpha.size

        if n == 0 or P_total <= 0 or not np.all(alpha > 0):
            return {"x": [float("nan")] * n, "Capacity": float("nan")}

        # Helper: compute sum of feasible x for a given lambda
        def _sum_x(lam: float) -> float:
            # x_i = max(1/lam - alpha_i, 0)
            vals = 1.0 / lam - alpha
            vals[vals < 0] = 0.0
            return vals.sum()

        # Find lambda via bisection over [lam_low, lam_high].
        # Since x_i decreases as lambda increases.
        lam_low = 1e-9            # yields huge x_i (overflow avoided)
        lam_high = 1e9            # yields near 0
        # Narrow the interval using checks
        if _sum_x(lam_low) < P_total:
            # too many resources, need smaller lambda
            lam_low = _find_lower_bound(alpha, P_total, lam_low)
        for _ in range(100):
            lam_mid = 0.5 * (lam_low + lam_high)
            s_mid = _sum_x(lam_mid)
            if s_mid > P_total:
                lam_low = lam_mid
            else:
                lam_high = lam_mid
        lam = 0.5 * (lam_low + lam_high)

        # Construct solution
        x = 1.0 / lam - alpha
        x[x < 0] = 0.0
        # Slight adjustment in case of numerical drift
        curr_sum = x.sum()
        if curr_sum > 0:
            x *= P_total / curr_sum

        # Compute capacity
        capacity = np.sum(np.log(alpha + x))

        return {"x": x.tolist(), "Capacity": float(capacity)}

def _find_lower_bound(alpha: np.ndarray, P_total: float, lam_start: float) -> float:
    """find a lower bound for lambda where sum_x < P_total"""
    lam = lam_start
    step = 2.0
    while True:
        # For very large lambda, sum_x ~ 0, so we need to expand lam downwards
        s = (1.0 / lam - alpha)
        s[s < 0] = 0.0
        if s.sum() < P_total:
            return lam
        lam *= step  # tighten lam