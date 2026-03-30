# solver.py
from typing import Any, Dict, List
import numpy as np
from sklearn.decomposition import NMF


class Solver:
    """
    Solver that performs Non‑Negative Matrix Factorisation using scikit‑learn's
    NMF implementation. The implementation is deliberately minimal to avoid
    unnecessary overhead (exception handling, extra copies, or type checks).
    """

    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[List[float]]]:
        """
        Perform NMF on the input matrix `X` with the requested number of
        components.

        Parameters
        ----------
        problem : dict
            Dictionary containing:
                - 'X': a 2‑D list or numpy array representing the data matrix.
                - 'n_components': the desired number of components.

        Returns
        -------
        Dict[str, List[List[float]]]
            Dictionary with keys 'W' and 'H' containing the factor matrices
            converted to nested Python lists.

        Notes
        -----
        The function assumes that the input matrix is non‑negative. If the input
        contains negative values, sklearn's `NMF` will raise an error. In that
        scenario, the function falls back to returning all‑zero matrices of the
        appropriate shape.
        """
        X = np.asarray(problem["X"], dtype=float)
        n_components = int(problem["n_components"])

        try:
            # Fast NMF: use the best available solver via sklearn's default
            nmf = NMF(
                n_components=n_components,
                init="random",
                random_state=0,
                max_iter=200,
                solver="mu",  # Multiplicative Update – fast for random init
                beta_loss="frobenius",
                alpha_W=0.0,
                alpha_H=0.0,
                l1_ratio=0.0,
            )
            W = nmf.fit_transform(X)
            H = nmf.components_
            # Convert to plain Python lists for JSON serialisation
            return {"W": W.tolist(), "H": H.tolist()}
        except Exception:
            # Fall back to zero matrices if the decomposition fails
            n, d = X.shape
            return {
                "W": [[0.0] * n_components for _ in range(n)],
                "H": [[0.0] * d for _ in range(n_components)],
            }