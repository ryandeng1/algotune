import numpy as np

def solve(problem: dict[str, list[list[float]]]) -> list[float]:
    """
    Return the two eigenvalues of the given symmetric matrix that are
    closest to zero by absolute value.
    """
    # Convert to a numpy array (dtype float32 for speed if not needed precision)
    matrix = np.asarray(problem["matrix"], dtype=np.float32, order="K")
    # Compute all eigenvalues (efficient for symmetric matrices)
    eig_vals = np.linalg.eigvalsh(matrix)
    # Find indices of the two smallest absolute eigenvalues without full sort
    idx_small = np.argpartition(np.abs(eig_vals), 2)[:2]
    # Extract and sort those two eigenvalues by absolute value
    res = eig_vals[idx_small]
    res_sorted = res[np.argsort(np.abs(res))]
    return res_sorted.tolist()