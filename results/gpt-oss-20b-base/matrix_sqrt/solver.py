# solver.py
import numpy as np
from scipy.linalg import sqrtm, eig

class Solver:
    def solve(self, problem, **kwargs):
        """
        Compute the principal matrix square root of a complex square matrix A.

        Parameters
        ----------
        problem : dict
            Expected to contain a key "matrix" whose value is a 2‑D array-like
            structure containing complex entries (strings or Python complex
            objects).  The entries may be nested lists or a NumPy array.

        Returns
        -------
        dict
            A dictionary with a single key "sqrtm" whose value is a dictionary
            containing the key "X".  The value of "X" is a list of lists of
            Python complex numbers that represent the principal square root.
        """
        # Convert input to a NumPy array of complex dtype
        A = np.array(problem["matrix"], dtype=complex)

        # If the matrix is tiny, fall back to SciPy's robust implementation
        if A.size <= 4:  # 2x2 or smaller
            X = sqrtm(A, disp=False)
        else:
            # Attempt eigen‑based square root. If it fails, use SciPy as a backup.
            try:
                # Compute eigenvalues and right eigenvectors
                w, v = eig(A)
                # Ensure we use the principal branch of sqrt
                sqrt_w = np.sqrt(w)
                # Construct X = V * diag(sqrt(w)) * V^{-1}
                X = v @ np.diag(sqrt_w) @ np.linalg.inv(v)
                # Verify the result; if error too large, fall back
                if not np.allclose(X @ X, A, rtol=1e-5, atol=1e-8):
                    X = sqrtm(A, disp=False)
            except Exception:
                # Any error (non‑diagonalizable, etc.) fallback
                X = sqrtm(A, disp=False)

        # Convert to serializable list of lists of Python complex numbers
        X_list = X.tolist()
        return {"sqrtm": {"X": X_list}}
