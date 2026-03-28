import numpy as np
from sklearn.svm import LinearSVC

class Solver:
    def solve(self, problem: dict) -> dict:
        """
        Solve a hard‑margin / soft‑margin linear SVM using scikit‑learn's LinearSVC.

        Parameters
        ----------
        problem : dict
            Must contain
            * 'X' : array‑like, shape (n_samples, n_features)
            * 'y' : array‑like, shape (n_samples,)
            * 'C' : float, regularisation parameter

        Returns
        -------
        dict
            * 'beta0'      : intercept (float)
            * 'beta'       : weight vector (list[float])
            * 'optimal_value' : objective value (float)
            * 'missclass_error' : classification error on training set (float)
        """
        X = np.asarray(problem["X"])
        y = np.asarray(problem["y"])
        C = problem.get("C", 1.0)

        # LinearSVC minimises (1/2)||w||^2 + C * Σ max(0, 1- y_i w^T x_i)
        # This is equivalent to the standard SVM formulation.
        clf = LinearSVC(C=C, loss="hinge", dual=False, fit_intercept=True, max_iter=10000, random_state=0)
        try:
            clf.fit(X, y)
        except Exception:
            return None

        beta = clf.coef_.flatten()
        beta0 = clf.intercept_[0]
        # compute objective value: 0.5 * ||beta||^2 + C * sum xi
        # with Dual slope: xi = max(0, 1 - y_i*(Xbeta + beta0))
        preds = X @ beta + beta0
        margins = y * preds
        xi = np.maximum(0, 1 - margins)
        optimal_value = 0.5 * np.sum(beta ** 2) + C * np.sum(xi)

        missclass_error = np.mean(preds * y < 0)

        return {
            "beta0": float(beta0),
            "beta": beta.tolist(),
            "optimal_value": float(optimal_value),
            "missclass_error": float(missclass_error),
        }