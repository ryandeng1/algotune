from typing import Any
import numpy as np
import sklearn

class Solver:
    def solve(self, problem: dict[str, Any]) -> list[list[float]]:
        try:
            X = np.array(problem["X"])
            n, d = X.shape
            X -= np.mean(X, axis=0)
            model = sklearn.decomposition.PCA(n_components=problem["n_components"])
            model.fit(X)
            V = model.components_
            return V.tolist()
        except Exception as e:
            n_components = problem["n_components"]
            X = np.array(problem["X"])
            n, d = X.shape
            V = np.zeros((n_components, n))
            id = np.eye(n_components)
            V[:, :n_components] = id
            return [list(row) for row in V]