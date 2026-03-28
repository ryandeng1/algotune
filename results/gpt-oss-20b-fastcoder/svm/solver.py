import numpy as np
from typing import Any, Dict, List

# Using scikit‑learn's fast LinearSVC which solves the hinge‑loss SVM in primal
# via a coordinate‑descent algorithm; it is orders of magnitude faster than
# convex‑optimization libraries for large problems.
try:
    from sklearn.svm import LinearSVC
except ImportError:  # pragma: no cover
    LinearSVC = None  # type: ignore


class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Solve a linear SVM using scikit‑learn's LinearSVC (C‑SVM).
        The returned dictionary contains:
            - beta0 : bias (float)
            - beta  : coefficient vector (list[float])
            - optimal_value : training objective value (float)
            - missclass_error : training error (float)
        """
        if LinearSVC is None:
            raise RuntimeError("scikit‑learn not available")

        X = np.asarray(problem["X"], dtype=float)
        y = np.asarray(problem["y"], dtype=float).ravel()
        C = float(problem["C"])

        # scikit‑learn expects labels to be {-1, 1}
        y = np.where(y <= 0, -1, 1)

        clf = LinearSVC(
            C=C,
            loss="hinge",
            fit_intercept=True,
            max_iter=10000,
            tol=1e-5,
            dual=False,
            verbose=0,
        )
        clf.fit(X, y)

        beta = clf.coef_.ravel()
        beta0 = clf.intercept_.item()

        # Compute objective value: 0.5 * ||beta||^2 + C * sum(max(0, 1 - y*(Xβ+β0))
        margins = y * (X @ beta + beta0)
        hinge_loss = np.maximum(0, 1 - margins).sum()
        optimal_value = 0.5 * np.dot(beta, beta) + C * hinge_loss

        # Training miss‑classification error
        predictions = X @ beta + beta0
        missclass_error = np.mean(predictions * y < 0)

        return {
            "beta0": float(beta0),
            "beta": beta.tolist(),
            "optimal_value": float(optimal_value),
            "missclass_error": float(missclass_error),
        }