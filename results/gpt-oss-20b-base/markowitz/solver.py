import numpy as np
from cvxopt import matrix, solvers

# Disable solver output for speed
solvers.options['show_progress'] = False
solvers.options['maxiters'] = 2000

def _qpsolve(μ: np.ndarray, Σ: np.ndarray, γ: float):
    """Solve the quadratic program:
          max μ'w - γ w^T Σ w
          s.t. sum(w) = 1, w >= 0
    using cvxopt which is much faster than cvxpy for this simple case."""
    n = μ.size
    P = matrix(2 * γ * Σ, tc='d')
    q = matrix(-μ, tc='d')                # objective: 0.5 w^T P w + q^T w

    G = matrix(-np.eye(n), tc='d')        # w >= 0  ->  -w <= 0
    h = matrix(np.zeros(n), tc='d')

    A = matrix(np.ones((1, n)), tc='d')   # sum(w) = 1
    b = matrix([1.0], tc='d')

    sol = solvers.qp(P, q, G, h, A, b)
    if sol['status'] != 'optimal':
        return None
    return np.array(sol['x']).reshape(-1)

class Solver:
    def solve(self, problem: dict[str, any]) -> dict[str, list[float]] | None:
        μ = np.asarray(problem['μ'], dtype=float)
        Σ = np.asarray(problem['Σ'], dtype=float)
        γ = float(problem['γ'])
        if μ.shape != Σ.shape[0:2] or Σ.shape[0] != Σ.shape[1]:
            return None
        w = _qpsolve(μ, Σ, γ)
        if w is None or not np.isfinite(w).all():
            return None
        return {'w': w.tolist()}