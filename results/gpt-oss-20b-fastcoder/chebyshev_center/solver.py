from typing import Any
import numpy as np
from scipy.optimize import linprog

def solve(problem: dict[str, Any]) -> dict[str, list]:
    """
    Solve the Chebyshev center problem using linear programming.

    :param problem: A dictionary containing the matrices 'a' and vector 'b'.
    :return: A dictionary with key "solution" containing the optimal point.
    """
    a = np.asarray(problem['a'], dtype=float)
    b = np.asarray(problem['b'], dtype=float)

    n = a.shape[1]
    # Norm of each row of a
    norms = np.linalg.norm(a, axis=1)

    # Objective: maximize r -> minimize -r
    c = np.zeros(n + 1)
    c[-1] = -1.0  # coefficient for r

    # Inequality constraints: a @ x + r * norms <= b
    A_ub = np.hstack([a, norms.reshape(-1, 1)])
    b_ub = b

    # All variables are free (unbounded)
    bounds = [(None, None)] * (n + 1)

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    if not res.success:
        raise RuntimeError(f"Linear program failed: {res.message}")

    x_opt = res.x[:-1]  # exclude r
    return {'solution': x_opt.tolist()}