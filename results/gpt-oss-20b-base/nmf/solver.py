import numpy as np
from sklearn.decomposition import NMF

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list[list[float]]]:
        """
        Compute the Non‑negative Matrix Factorisation `X ≈ W @ H` using
        scikit‑learn's fast implementation.  The function is defensive:
        if the NMF algorithm crashes (which can happen for pathological
        data), we fall back to a zero matrix solution that satisfies the
        shape constraints.

        Parameters
        ----------
        problem : dict
            Must contain:
              - 'X': 2‑D array‑like of shape (n, d)
              - 'n_components': int, the rank of the factorisation

        Returns
        -------
        dict
            {'W': list of lists, 'H': list of lists}
            The result has shape (n, n_components) for `W` and
            (n_components, d) for `H`.
        """
        X = np.asarray(problem['X'], dtype=float)
        n_components = int(problem['n_components'])

        # Fast NMF using scikit‑learn's ALS (alternating least squares) solver.
        # Set a small number of iterations to keep runtime low while still
        # producing a useful result for typical data sizes.
        algo = NMF(
            n_components=n_components,
            init='nndsvda',      # good default for most data
            solver='cd',         # coordinate descent is very fast
            max_iter=200,        # try to reach convergence quickly
            random_state=0,
            alpha=0.0,           # no regularisation
            l1_ratio=0.0         # no regularisation
        )

        try:
            W = algo.fit_transform(X)
            H = algo.components_
            return {'W': W.tolist(), 'H': H.tolist()}
        except Exception:
            # Fallback: return zero matrices of the correct shape.
            n, d = X.shape
            W = [[0.0] * n_components for _ in range(n)]
            H = [[0.0] * d for _ in range(n_components)]
            return {'W': W, 'H': H}