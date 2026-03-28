import numpy as np
from typing import Any, List

class Solver:
    def solve(self, problem: dict[str, Any]) -> List[List[float]]:
        X = np.asarray(problem["X"], dtype=np.float64)
        n_components = int(problem["n_components"])

        # Center the data
        X -= X.mean(axis=0)

        # Perform thin SVD
        U, _, Vt = np.linalg.svd(X, full_matrices=False)

        # Take the first n_components principal components
        components = Vt[:n_components]

        # Convert to list of lists (float) for consistency with expected output
        return components.tolist()