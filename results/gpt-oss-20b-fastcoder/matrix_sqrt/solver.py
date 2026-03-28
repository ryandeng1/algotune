from typing import Any, Dict, List
import numpy as np

def solve(problem: Dict[str, np.ndarray]) -> Dict[str, Dict[str, List[List[complex]]]]:
    """
    Fast principal matrix square root via eigen‑decomposition.

    Parameters
    ----------
    problem : dict
        Dictionary with key ``"matrix"`` holding a square NumPy array.

    Returns
    -------
    dict
        Dictionary with key ``"sqrtm"`` that contains a dictionary with key
        ``"X"`` – list of lists of complex numbers representing the
        principal square‑root matrix.
    """
    A: np.ndarray = problem["matrix"]

    # Ensure we are working with a NumPy array with sufficient precision.
    A = np.asarray(A, dtype=np.complex128)

    # Attempt eigen‑decomposition.
    try:
        w, v = np.linalg.eig(A)
    except np.linalg.LinAlgError:
        return {"sqrtm": {"X": []}}

    # Principal square roots of eigenvalues.
    sqrt_w = np.sqrt(w)

    # Construct the square‑root matrix.
    # X = V @ diag(sqrt_w) @ V^{-1}
    # Use solve instead of explicit inverse for numerical stability.
    try:
        X = v @ np.diag(sqrt_w)
        X = np.linalg.solve(v, X)
    except np.linalg.LinAlgError:
        return {"sqrtm": {"X": []}}

    # Convert to list of lists of Python complex numbers.
    X_list = X.tolist()

    return {"sqrtm": {"X": X_list}}