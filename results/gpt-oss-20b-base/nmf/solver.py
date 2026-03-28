import numpy as np
from sklearn.decomposition import NMF

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list[list[float]]]:
        X = np.asarray(problem["X"], dtype=float)
        n_components = int(problem["n_components"])
        # Perform NMF with deterministic init for reproducibility
        nmf_model = NMF(n_components=n_components, init="random", random_state=0, max_iter=200)
        W = nmf_model.fit_transform(X)
        H = nmf_model.components_
        # Convert to nested Python lists only once
        return {"W": W.tolist(), "H": H.tolist()}