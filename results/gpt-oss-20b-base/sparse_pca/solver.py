import numpy as np

def solve(problem: dict) -> dict:
    """
    Fast, approximate sparse PCA solver.
    This implementation takes the top eigenvectors of the covariance matrix,
    then promotes sparsity by keeping only the largest |α| entries of each component,
    where α is a user‑defined sparsity parameter (between 0 and 1).

    :param problem: Dictionary with problem parameters
    :returns: Dictionary with the sparse principal components and their explained variance
    """
    # Get inputs
    A = np.asarray(problem["covariance"], dtype=float)
    n_components = int(problem.get("n_components", 2))
    sparsity_param = float(problem.get("sparsity_param", 0.1))

    # Compute eigen‑decomposition
    eigvals, eigvecs = np.linalg.eigh(A)

    # Keep only positive eigenvalues
    pos = eigvals > 0
    eigvals = eigvals[pos]
    eigvecs = eigvecs[:, pos]

    # Sort in decreasing eigenvalue order
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Number of components to retain
    k = min(len(eigvals), n_components)

    # Build the initial dense components
    components = eigvecs[:, :k] * np.sqrt(eigvals[:k])

    # Sparsity step: keep only the largest |α| fraction of elements in each component,
    # then renormalise to unit length (to mimic the constraints in the original CVXPY model)
    n, _ = components.shape
    for j in range(k):
        col = components[:, j]
        # Determine how many elements to keep
        keep = max(1, int(np.ceil(sparsity_param * n)))
        # Zero all but the keep largest absolute values
        thresh = np.partition(np.abs(col), -keep)[-keep]
        col[np.abs(col) < thresh] = 0.0
        # Normalise to unit norm
        norm = np.linalg.norm(col)
        if norm > 0:
            components[:, j] = col / norm

    # Compute explained variance for each component
    explained_variance = []
    for j in range(k):
        v = components[:, j]
        explained_variance.append(float(v.T @ A @ v))

    return {"components": components.tolist(), "explained_variance": explained_variance}