import numpy as np
import sklearn

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list[list[float]]]:
        X = np.array(problem["X"])
        n_components = problem["n_components"]
        model = sklearn.decomposition.NMF(
            n_components=n_components,
            init="random",
            random_state=0
        )
        W = model.fit_transform(X)
        H = model.components_
        return {"W": W.tolist(), "H": H.tolist()}
