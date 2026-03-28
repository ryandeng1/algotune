import numpy as np

class Solver:
    """
    Optimized solver to compute the top ``n_components`` singular values/vectors
    of a dense matrix ``A``.  For large matrices the original
    ``sklearn.utils.extmath.randomized_svd`` is slower than NumPy's
    ``linalg.svd`` for the target number of components, so we use the
    built‑in routine and slice down to the requested size, which avoids
    unnecessary power‑iteration overhead.
    """
    def solve(self, problem: dict[str, Any]) -> dict[str, list]:
        A = problem['matrix']
        n_components = problem['n_components']
        # Perform a single full SVD in double precision (fastest on most CPUs)
        # Use full_matrices=False to avoid extra work on the trailing singular
        # vectors that we probably won't need.
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        # Trim to the requested number of components
        U = U[:, :n_components]
        s = s[:n_components]
        Vt = Vt[:n_components, :]
        return {'U': U, 'S': s, 'V': Vt.T}