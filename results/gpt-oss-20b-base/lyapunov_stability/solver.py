# Python 3.10
import numpy as np
import scipy.linalg as la

def solve(problem: dict):
    """
    Solves the discrete Lyapunov stability analysis problem
    by checking stability analytically and, if stable,
    solving the discrete Lyapunov equation:

        P - A.T @ P @ A = I.

    Args:
        problem: dictionary with key 'A' containing a square matrix

    Returns:
        dict with keys:
            'is_stable': (bool) whether the system is asymptotically stable,
            'P': (list[list[float]]) the Lyapunov matrix if stable, else None
    """
    A = np.atleast_2d(problem['A']).astype(float)
    n = A.shape[0]
    assert A.shape[1] == n, "A must be square"

    # 1. Check asymptotic stability: all eigenvalues inside unit circle
    eig_vals = np.linalg.eigvals(A)
    if np.any(np.abs(eig_vals) >= 1):
        return {'is_stable': False, 'P': None}

    # 2. Solve the discrete Lyapunov equation
    try:
        P = la.solve_discrete_lyapunov(A.T, np.eye(n))
        # Ensure symmetry
        P = (P + P.T) / 2.0
        return {'is_stable': True, 'P': P.tolist()}
    except Exception:
        # Fallback if Lyapunov solver fails
        return {'is_stable': False, 'P': None}