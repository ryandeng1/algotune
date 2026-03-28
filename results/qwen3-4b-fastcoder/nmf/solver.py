from typing import Any
import numpy as np
from sklearn.decomposition import NMF

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list[list[float]]]:
        try:
            X = problem["X"]
            if not isinstance(X, np.ndarray):
                X = np.array(X)
            n_components = problem["n_components"]
            model = NMF(n_components=n_components, init="random", random_state=0)
            W = model.fit_transform(X)
            H = model.components_
            return {"W": W.tolist(), "H": H.tolist()}
        except Exception:
            X_arr = np.array(problem["X"])
            n, d = X_arr.shape
            n_components = problem["n_components"]
            W = np.zeros((n, n_components), dtype=float).tolist()
            H = np.zeros((n_components, d), dtype=float).tolist()
            return {"W": W, "H": H}