import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

def solve(problem: dict) -> list[float]:
    """Return the k smallest eigenvalues of a symmetric matrix."""
    mat = problem["matrix"].asformat("csr")
    k = int(problem["k"])
    n = mat.shape[0]

    # If k is large or matrix is small, use dense computation
    if k >= n or n < 2 * k + 1:
        vals = np.linalg.eigvalsh(mat.toarray())
        return [float(v) for v in vals[:k]]

    # Otherwise use Lanczos via eigsh
    try:
        vals = eigsh(mat,
                     k=k,
                     which="SM",
                     return_eigenvectors=False,
                     maxiter=n * 200,
                     ncv=min(n - 1, max(2 * k + 1, 20)))[0]
    except Exception:
        vals = np.linalg.eigvalsh(mat.toarray())[:k]

    return [float(v) for v in np.sort(np.real(vals))]