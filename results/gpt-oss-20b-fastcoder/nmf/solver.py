import numpy as np
import sklearn.decomposition

class Solver:
    def solve(self, problem: dict[str, any]) -> dict[str, list[list[float]]]:
        # Convert to numpy array once
        X = np.asarray(problem["X"], dtype=float)
        n_components = problem["n_components"]

        # Perform NMF using scikit‑learn
        model = sklearn.decomposition.NMF(
            n_components=n_components, init="random", random_state=0, max_iter=200
        )
        W = model.fit_transform(X)
        H = model.components_

        # Convert the results to plain Python lists
        return {"W": W.tolist(), "H": H.tolist()}