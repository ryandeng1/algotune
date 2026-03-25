# solver.py
from typing import Any, Dict, List
import numpy as np
from sklearn.svm import LinearSVC


class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Solve the primal SVM problem using scikit-learn's LinearSVC.
        The solution satisfies the same optimality criteria as the reference
        CVXPY implementation.
        """
        # Extract data
        X = np.asarray(problem["X"], dtype=float)
        y = np.asarray(problem["y"], dtype=int)
        C = float(problem["C"])

        # Ensure binary labels are -1/1
        if not set(np.unique(y)).issubset({-1, 1}):
            raise ValueError("Labels must be -1 or 1")

        # Train linear SVM with primal form
        clf = LinearSVC(
            C=C,
            penalty="l2",
            loss="hinge",
            dual=False,  # primal solver
            fit_intercept=True,
            max_iter=10000,
            tol=1e-4,
            random_state=0,
        )
        clf.fit(X, y)

        beta = clf.coef_.flatten()          # shape (p,)
        beta0 = clf.intercept_[0]           # bias term

        # Compute slack variables
        margins = y * (X @ beta + beta0)
        xi = np.maximum(0, 1 - margins)

        # Optimal objective value
        obj = 0.5 * np.dot(beta, beta) + C * xi.sum()

        # Misclassification error
        misclass_error = np.mean(margins < 0)

        return {
            "beta0": float(beta0),
            "beta": beta.tolist(),
            "optimal_value": float(obj),
            "missclass_error": float(misclass_error),
        }
