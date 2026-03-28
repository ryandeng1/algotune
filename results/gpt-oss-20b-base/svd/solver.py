import numpy as np
from typing import Any, Dict, List

def solve(problem: Dict[str, Any]) -> Dict[str, List]:
    """
    Compute the singular value decomposition of a real matrix A
    using NumPy's accelerated routines.

    Parameters
    ----------
    problem : dict
        Dictionary containing the key 'matrix' mapping to a 2‑D iterable.

    Returns
    -------
    dict
        Dictionary with keys
        'U' : list[list[float]] – left singular vectors,
        'S' : list[float]      – singular values,
        'V' : list[list[float]] – right singular vectors.
    """
    # Ensure the input is a real‐valued dense array; contiguous
    A = np.ascontiguousarray(problem["matrix"], dtype=np.float64)

    # Fast SVD; no full matrices, no unnecessary copy
    U, s, Vh = np.linalg.svd(A, full_matrices=False, compute_uv=True)

    # Convert to Python lists for the required API
    return {
        "U": U.tolist(),
        "S": s.tolist(),
        "V": Vh.T.tolist(),
    }