from typing import Any, Dict, List
import numpy as np
import cvxpy as cp

def solve(problem: Dict[str, Any]) -> Dict[str, List[float]]:
    """
    Solve the LP‑centering problem using CVXPY with a fast interior‑point solver.

    Parameters
    ----------
    problem : dict
        Dictionary containing:
            * 'c' : array‑like of shape (n,)
            * 'A' : array‑like of shape (m, n)
            * 'b' : array‑like of shape (m,)

    Returns
    -------
    dict
        {'solution': list[float]} containing the optimal x.
    """
    # Convert inputs to Numpy arrays for speed
    c = np.asarray(problem["c"], dtype=np.float64)
    A = np.asarray(problem["A"], dtype=np.float64)
    b = np.asarray(problem["b"], dtype=np.float64)

    n = c.shape[0]
    x = cp.Variable(n, name="x")

    # Objective: minimize cᵀx – Σ log(x_i)
    obj = cp.Minimize(c @ x - cp.sum(cp.log(x)))
    constraints = [A @ x == b]
    prob = cp.Problem(obj, constraints)

    # Use a fast, reliable solver (SCS) with minimal tolerance for speed
    prob.solve(solver=cp.SCS,
               verbose=False,
               eps=1e-8,
               max_iters=5000)

    assert prob.status == "optimal", f"Solver status: {prob.status}"
    return {"solution": x.value.tolist()}