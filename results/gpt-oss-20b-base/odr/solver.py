import numpy as np

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """Fit weighted ODR (approximate) using iterative weighted least squares."""
        x = np.asarray(problem["x"], dtype=float)
        y = np.asarray(problem["y"], dtype=float)
        sx = np.asarray(problem["sx"], dtype=float)
        sy = np.asarray(problem["sy"], dtype=float)

        # Prepare design matrix
        X = np.column_stack((x, np.ones_like(x)))

        # Initial guess using ordinary weighted least squares (only y errors)
        w = 1.0 / (sy**2 + 1e-15)
        beta = np.linalg.lstsq(X * np.sqrt(w[:, None]), y * np.sqrt(w), rcond=None)[0]

        # Iteratively refine weights accounting for x errors
        for _ in range(5):
            slope = beta[0]
            # Effective variance of orthogonal residuals
            sigma2 = sx**2 * slope**2 + sy**2
            w = 1.0 / (sigma2 + 1e-15)
            beta = np.linalg.lstsq(X * np.sqrt(w[:, None]), y * np.sqrt(w), rcond=None)[0]

        return {"beta": beta.tolist()}