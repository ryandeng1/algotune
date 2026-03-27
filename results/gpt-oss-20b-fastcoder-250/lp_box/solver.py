from __future__ import annotations
from typing import Any, Dict, List
import numpy as np
from scipy.optimize import linprog


def solve(problem: Dict[str, Any]) -> Dict[str, List[float]]:
    """
    Solve the LP box problem using SciPy's highly‑efficient linprog implementation instead of CVXPY.

    Parameters
    ----------
    problem : dict
        Dictionary containing keys:
            "c" : iterable of coefficient of the objective.
            "A" : 2D iterable of constraint matrix (rows of inequalities A @ x <= b).
            "b" : iterable of right‑hand‑side of inequalities.

    Returns
    -------
    dict
        Dictionary with key 'solution' mapping to the optimal 1‑D list of decision variables.
    """
    # Extract problem data as NumPy arrays
    c = np.asarray(problem["c"], dtype=float)
    A = np.asarray(problem["A"], dtype=float)
    b = np.asarray(problem["b"], dtype=float)

    # Bounds: 0 <= x_i <= 1 for all variables
    bounds = [(0.0, 1.0)] * c.size

    # Call SciPy's linprog.  In large problems 'highs' (default) is fast and robust.
    res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs")

    # Raise an error if the problem is infeasible or unbounded.
    if not res.success:
        raise ValueError(f"LP solver failed: {res.message}")

    # Ensure that the solution is cast to a plain list of Python floats.
    return {"solution": res.x.tolist()}