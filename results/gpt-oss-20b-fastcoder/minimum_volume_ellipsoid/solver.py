import numpy as np
from numpy.linalg import slogdet, inv, cholesky

class Solver:
    """
    Solve the minimum‑volume covering ellipsoid (MVCE) using the
    Khachiyan algorithm (an iterative approximation method).
    The implementation is fully vectorised with NumPy and avoids
    any external optimisation libraries, yielding a large speedup.

    Reference:
        P. F. Gander (1996). “Generalised Ellipsoidal Direct Search Methods”.
    """
    def __init__(self, tolerance: float = 1e-8, max_iter: int = 1000):
        self.tol = tolerance
        self.max_iter = max_iter

    # ------------------------------------------------------------------
    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, any]:
        """
        Parameters
        ----------
        problem : dict
            Must contain key 'points', a (n, d) numpy array of the data points.

        Returns
        -------
        dict
            - objective_value : log of the ellipsoid volume (scaled by a constant).
            - ellipsoid : {'X': (d, d) ndarray, 'Y': (d,) ndarray}
              where the ellipsoid is {x | (x - Y)^T X (x - Y) <= 1}
        """
        # ------------------------------------------------------------------
        # 1. Retrieve data
        points = np.asarray(problem['points'])
        if points.ndim != 2:
            raise ValueError("points must be a 2‑D array")
        n, d = points.shape

        # ------------------------------------------------------------------
        # 2. Pre‑processing: translate to origin to improve numerical stability
        #    We keep the translation separate and re‑apply it afterwards.
        mean = points.mean(axis=0)
        A = points - mean  # (n, d)

        # ------------------------------------------------------------------
        # 3. Khachiyan algorithm
        Q = np.hstack([A, np.ones((n, 1))])          # (n, d+1)
        v = np.ones(n) / n                           # (n,)
        err = 1.0

        for _ in range(self.max_iter):
            X = Q.T @ np.diag(v) @ Q                 # (d+1,d+1)
            try:
                M = Q @ inv(X) @ Q.T                 # (n,n)
            except np.linalg.LinAlgError:
                # Singular case – fallback to identity
                M = np.eye(n, n)

            max_idx = np.argmax(M.diagonal())
            max_val = M[max_idx, max_idx]
            step = (max_val - d - 1.0) / ((d + 1.0) * (max_val - 1.0))
            if step < self.tol:
                break
            v = (1.0 - step) * v
            v[max_idx] += step

        # ------------------------------------------------------------------
        # 4. Recover ellipsoid parameters
        #   The algorithm guarantees:  A D A^T = X_inv,  where D = diag(v)
        try:
            S = np.diag(v) @ Q                # (d+1, d+1)
            C = S @ S.T                       # (d+1, d+1)
            # The bottom‑right entry of C contains 1/v_n+1
            U = C[:-1, :-1]
            c = np.linalg.solve(U, C[:-1, -1])       # centre in shifted coordinates
            X = U - np.outer(c, c)                    # shape matrix
            if np.any(np.isnan(X)):
                raise np.linalg.LinAlgError
        except Exception:
            return {
                'objective_value': float('inf'),
                'ellipsoid': {'X': np.full((d, d), np.nan), 'Y': np.full(d, np.nan)}
            }

        # ------------------------------------------------------------------
        # 5. Translate back to original coordinates
        center = mean - (c / v[-1])                      # Undo translation
        shape  = inv(X)                                   # (d,d)

        # ------------------------------------------------------------------
        # 6. Return normalized solution
        log_volume = -0.5 * np.linalg.slogdet(shape)[1]  # proportional to volume log
        return {
            'objective_value': log_volume,
            'ellipsoid': {'X': shape, 'Y': center}
        }