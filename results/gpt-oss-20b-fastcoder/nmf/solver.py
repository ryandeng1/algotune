from typing import Any, Dict, List

def solve(problem: dict[str, Any]) -> dict[str, List[List[float]]]:
    """
    Fast non-negative matrix factorisation using multiplicative updates.
    Implements a simple and efficient version of NMF that runs entirely with NumPy.
    """
    import numpy as np

    X = np.array(problem["X"], dtype=float)

    n, d = X.shape
    n_components = problem["n_components"]

    # initialise W and H with suitable non‑negative values
    rng = np.random.default_rng(seed=0)
    W = rng.uniform(low=0.0, high=1.0, size=(n, n_components))
    H = rng.uniform(low=0.0, high=1.0, size=(n_components, d))

    # number of iterations – a small fixed number gives a quick result
    n_iter = 50
    eps = 1e-9

    for _ in range(n_iter):
        # Update H
        WH = W @ H
        H *= (W.T @ X) / (W.T @ WH + eps)
        # Update W
        WH = W @ H
        W *= (X @ H.T) / (WH @ H.T + eps)

    return {"W": W.tolist(), "H": H.tolist()}