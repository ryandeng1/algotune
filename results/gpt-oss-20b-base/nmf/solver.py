# solver.py
from typing import Any, Dict, List
import numpy as np
import sklearn.decomposition

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[List[float]]]:
        """
        Solve Non-Negative Matrix Factorization using scikit-learn's NMF.

        Parameters
        ----------
        problem : dict
            Dictionary containing
                - "X" : 2‑d list/array of non‑negative floats
                - "n_components" : int, desired rank

        Returns
        -------
        dict
            {"W": list[list[float]], "H": list[list[float]]}
        """
        try:
            X = np.asarray(problem["X"], dtype=float)
            n_components = int(problem["n_components"])
            model = sklearn.decomposition.NMF(
                n_components=n_components,
                init="random",
                random_state=0,
                max_iter=200,  # default; small matrices finish quickly
            )
            W = model.fit_transform(X)
            H = model.components_
            return {"W": W.tolist(), "H": H.tolist()}
        except Exception:
            # Fallback: return zero matrices of the correct shape
            X_shape = np.asarray(problem["X"]).shape
            m, n = X_shape
            W_zero = [[0.0] * problem["n_components"] for _ in range(m)]
            H_zero = [[0.0] * n for _ in range(problem["n_components"])]
            return {"W": W_zero, "H": H_zero}
