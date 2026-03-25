import numpy as np
from scipy.optimize import linprog
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, Any]:
        X = np.asarray(problem["X"], dtype=float)
        y = np.asarray(problem["y"], dtype=float)
        q = float(problem["quantile"])
        fit_int = bool(problem["fit_intercept"])

        n_samples, n_features = X.shape
        n_vars = n_features + (1 if fit_int else 0) + 2 * n_samples

        # Objective: minimize sum(q * u_i + (1-q) * v_i)
        c = np.zeros(n_vars, dtype=float)
        if fit_int:
            betas_start = 0
            inter_start = n_features
            uv_start = n_features + 1
        else:
            betas_start = 0
            inter_start = None
            uv_start = n_features
        c[uv_start:uv_start + n_samples] = q          # u_i
        c[uv_start + n_samples:] = 1 - q              # v_i

        # Equality constraints: y_i - X_i * beta - intercept (if any) = u_i - v_i
        A_eq = np.empty((n_samples, n_vars), dtype=float)
        A_eq.fill(0.0)

        # Beta coefficients
        A_eq[:, betas_start:betas_start + n_features] = -X

        # Intercept coefficient
        if fit_int:
            A_eq[:, inter_start] = -1.0

        # u_i and v_i coefficients
        A_eq[:, uv_start:uv_start + n_samples] = 1.0          # u_i
        A_eq[:, uv_start + n_samples:] = -1.0                 # -v_i

        b_eq = y

        # Bounds: beta and intercept unrestricted; u_i, v_i ≥ 0
        bounds = []
        for _ in range(n_features):
            bounds.append((None, None))
        if fit_int:
            bounds.append((None, None))
        for _ in range(2 * n_samples):
            bounds.append((0, None))

        res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
        if not res.success:
            raise RuntimeError(f"Linear program failed: {res.message}")

        beta = res.x[betas_start:betas_start + n_features]
        intercept = np.array([res.x[inter_start]]) if fit_int else np.array([0.0])

        preds = X @ beta + (intercept if fit_int else 0.0)
        return {
            "coef": beta.reshape(1, -1).tolist(),
            "intercept": intercept.tolist(),
            "predictions": preds.tolist(),
        }
