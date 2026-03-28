import numpy as np
import cvxpy as cp

def solve(problem: dict[str, list[list[int]] | list[float] | int]) -> dict[str, list[list[float]] | float]:
    """
    Solve the Perron‑Frobenius matrix completion.

    Parameters
    ----------
    problem
        Dictionary containing:
        - 'inds': list of known index pairs
        - 'a'   : corresponding known values
        - 'n'   : dimension of the matrix

    Returns
    -------
    dict
        Dictionary with keys
        - 'B'            : completed matrix (as Python list of lists)
        - 'optimal_value': optimum objective value

        Returns ``None`` if the problem is infeasible or solver errors.
    """
    # --- pre‑processing ----------------------------------------------------
    n = problem['n']
    inds = np.asarray(problem['inds'], dtype=int)
    a = np.asarray(problem['a'], dtype=float)

    # grab all indices in row‑major order
    rows, cols = np.indices((n, n)).reshape(2, -1)
    all_idx = np.vstack((rows, cols)).T

    # exclude known entries
    known_mask = np.zeros(all_idx.shape[0], dtype=bool)
    for (r, c) in inds:
        known_mask |= (all_idx[:, 0] == r) & (all_idx[:, 1] == c)
    other_idx = all_idx[~known_mask]

    # --- CVXPY formulation -----------------------------------------------
    B = cp.Variable((n, n), pos=True)

    constraints = [
        cp.prod(B[other_idx[:, 0], other_idx[:, 1]]) == 1.0,
        B[inds[:, 0], inds[:, 1]] == a
    ]

    objective = cp.Minimize(cp.pf_eigenvalue(B))
    problem_cvx = cp.Problem(objective, constraints)

    # --- solve -------------------------------------------------------------
    try:
        result = problem_cvx.solve(gp=True, verbose=False)
    except (cp.SolverError, Exception):
        return None

    if B.value is None:
        return None

    return {"B": B.value.tolist(), "optimal_value": float(result)}