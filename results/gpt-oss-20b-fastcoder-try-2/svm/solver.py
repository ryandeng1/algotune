from typing import Any, Dict, List
import numpy as np
from sklearn.svm import SVC

class Solver:
    """
    Fast linear SVM solver using scikit‑learn's SVC (LIBSVM backend).
    """

    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        # Parse input
        X = np.asarray(problem['X'], dtype=np.float64)
        y = np.asarray(problem['y'], dtype=np.float64).ravel()
        # In scikit‑learn target labels must be -1/1
        y = np.where(y > 0, 1.0, -1.0)

        C = float(problem['C'])

        # Use LIBSVM linear kernel (C solver) for speed
        clf = SVC(kernel='linear', C=C, probability=False, decision_function_shape='ovr')
        clf.fit(X, y)

        # Retrieve weight vector and intercept
        # For a single class problem we directly have coef_ and intercept_
        beta = clf.coef_.astype(np.float64).reshape(-1)
        beta0 = float(clf.intercept_[0])

        # Compute optimal objective value (approximated)
        # Dual objective: sum(alpha_i) - 0.5 * sum_i,j alpha_i alpha_j y_i y_j <x_i, x_j>
        # We can reuse dual_coef_ and support_vectors_
        coeff = clf.dual_coef_.astype(np.float64).ravel()          # α_i * y_i
        sv = clf.support_vectors_.astype(np.float64)              # support vectors
        svy = y[clf.support_]                                    # labels of support vectors
        inv_C = 1.0 / C
        support_dual = coeff   # α_i
        # Primal objective (0.5||w||^2 + C sum_xi)
        # We approximate using dual objective (should be equal)
        dual_obj = support_dual.sum() - 0.5 * (support_dual @ (sv @ sv.T @ support_dual))
        dual_obj = float(dual_obj)

        # Misclassification error on training data
        decision = clf.decision_function(X)
        missclass = float((decision * y < 0).mean())

        return {
            'beta0': beta0,
            'beta': beta.tolist(),
            'optimal_value': dual_obj,
            'missclass_error': missclass,
        }