from typing import Any
import numpy as np
from sklearn.decomposition import NMF

class Solver:

    def solve(self, problem: dict[str, Any]) -> dict[str, list[list[float]]]:
        # Extract problem data
        X = np.asarray(problem["X"], dtype=float)
        n_components = problem["n_components"]

        # Run NMF using sklearn's efficient implementation
        model = NMF(n_components=n_components, init="random", random_state=0, max_iter=500, tol=1e-4)
        W = model.fit_transform(X)
        H = model.components_

        # Convert to nested Python lists for the required output format
        return {"W": W.tolist(), "H": H.tolist()}