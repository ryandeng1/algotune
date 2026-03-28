import math
import numpy as np

def solve(problem: dict) -> dict:
    """
    Blahut‑Arimoto algorithm for maximizing the mutual information of a discrete memoryless channel.
    The input `P` should be a 2‑D list or array with shape (m, n) where
    * m : number of output symbols
    * n : number of input symbols
    p(y|x) is stored column‑wise:  P[y, x]
    """
    # Convert to a NumPy array and validate shape
    P = np.asarray(problem.get('P', None), dtype=float)
    if P.ndim != 2:
        return None
    m, n = P.shape
    if m <= 0 or n <= 0:
        return None

    # Small constant to avoid log(0)
    eps = 1e-15

    # Initial input distribution: uniform
    x = np.full(n, 1.0 / n, dtype=float)

    # Pre‑compute log of P to avoid repeated logs
    logP = np.log(P + eps)

    # Main loop of Blahut‑Arimoto
    for _ in range(1000):
        # Joint distribution p(y, x)
        p_xy = P * x[np.newaxis, :]          # shape (m, n)

        # Marginal distribution p(y)
        p_y = p_xy.sum(axis=1, keepdims=True)  # shape (m, 1)

        # Compute the auxiliary quantity: log(Q(y|x))
        # Q(y|x) = P(y|x) / p(y)
        logQ = logP - np.log(p_y + eps)       # shape (m, n)

        # Update rule in logarithmic domain to maintain numerical stability
        # log(x_new[i]) = sum_y P(y|i)*logQ[y, i] + log(x[i])
        ax = np.sum(P * logQ, axis=0) + np.log(x + eps)

        # Exponentiate and renormalize
        x_new = np.exp(ax)
        x_new /= x_new.sum()

        # Convergence check (relative change in the distribution)
        if np.linalg.norm(x_new - x, ord=1) < 1e-8:
            x = x_new
            break
        x = x_new

    # Final mutual information computation:
    # I = sum_{i, y} x_i * P(y|i) * (log2 P(y|i) - log2 p_y)
    joint = P * x[np.newaxis, :]
    I = np.sum(joint * (logP - np.log(p_y + eps))) / math.log(2.0)

    return {'x': x.tolist(), 'C': float(I)}