from typing import Any

def solve(problem: dict[str, Any]) -> dict[str, list[list[float]]]:
    import numpy as np

    # Parameters
    n_components = problem['n_components']
    X = np.asarray(problem['X'], dtype=float)
    n, d = X.shape

    # Simple multiplicative‑update NMF
    np.random.seed(0)
    W = np.abs(np.random.randn(n, n_components))
    H = np.abs(np.random.randn(n_components, d))

    eps = 1e-10
    for _ in range(50):                      # fixed iterations
        # update H
        WH = W @ H
        H *= (W.T @ X) / (W.T @ WH + eps)
        # update W
        WH = W @ H
        W *= (X @ H.T) / (WH @ H.T + eps)

    return {'W': W.tolist(), 'H': H.tolist()}