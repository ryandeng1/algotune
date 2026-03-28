from typing import Any
import numpy as np
import sklearn

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list[list[float]]]:
        X = np.array(problem["X"])
        try:
            model = sklearn.decomposition.NMF(
                n_components=problem["n_components"], init="random", random_state=0
            )
            W = model.fit_transform(X)
            H = model.components_
            return {"W": W.tolist(), "H": H.tolist()}
        except Exception as e:
            n, d = X.shape
            n_components = problem["n_components"]
            W = np.zeros((n, n_components), dtype=float).tolist()
            H = np.zeros((n_components, d), dtype=float).tolist()
            return {"W": W, "H": H}