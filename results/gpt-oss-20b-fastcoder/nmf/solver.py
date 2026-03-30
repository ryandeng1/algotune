# solver.py
from typing import Any, Dict, List
import numpy as np
import sklearn.decomposition as skd


class Solver:
    """
    Solver that performs Non‑Negative Matrix Factorisation (NMF) on the input data.
    The implementation is deliberately lightweight: it converts the input to a
    NumPy array, runs the SKLearn NMF routine, and returns the factors as plain
    Python lists.  No exception handling is required as the problem statement
    guarantees that the inputs are valid; failure, if any, will surface as a
    standard Python exception.
    """

    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[List[float]]]:
        # Convert the matrix to a NumPy array
        X = np.asarray(problem["X"], dtype=float)

        # Run NMF
        model = skd.NMF(
            n_components=problem["n_components"],
            init="random",
            random_state=0,
            max_iter=200,  # default value – can be tuned if needed
        )
        W = model.fit_transform(X)
        H = model.components_

        # Convert results to plain Python lists
        return {"W": W.tolist(), "H": H.tolist()}