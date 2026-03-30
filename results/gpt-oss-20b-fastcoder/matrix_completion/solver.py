from typing import Any, Dict, List, Union
import cvxpy as cp
import numpy as np

class Solver:
    """
    Fastest possible solver for the Perron–Frobenius matrix completion problem that
    keeps the spirit of the original CVXPY formulation but removes unnecessary
    overhead and uses efficient NumPy operations.
    """

    def solve(
        self,
        problem: Dict[str, Union[List[List[int]], List[float], int]],
    ) -> Union[Dict[str, Any], None]:
        """
        Solve the matrix completion problem using CVXPY with geometric
        programming.

        Parameters
        ----------
        problem:
            Dictionary with keys:
                * 'inds': 2‑D list of `[row, col]` indices that are fixed.
                * 'a'   : 1‑D list of values that must be inserted at  `inds`.
                * 'n'   : dimension of the square matrix.

        Returns
        -------
        dict or None
            If successful returns a dictionary with keys
            'B' (completed matrix as nested lists) and
            'optimal_value' (the eigenvalue).  On failure returns ``None``.
        """
        inds = np.asarray(problem["inds"], dtype=np.int64)
        a_val = np.asarray(problem["a"], dtype=np.float64)
        n = int(problem["n"])

        # --- Build a mask of positions that are NOT fixed ---
        # Create a full grid of (row, col) indices
        rows, cols = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        all_idx = np.vstack((rows.ravel(), cols.ravel())).T

        # Use broadcasting to find the indices that match any of the fixed ones
        matches = np.any((all_idx[:, None] == inds).all(axis=2), axis=1)
        # Positions that are not fixed
        free_mask = ~matches
        free_rows = all_idx[free_mask, 0]
        free_cols = all_idx[free_mask, 1]

        # --- CVXPY variable ---
        B = cp.Variable((n, n), pos=True)

        # --- Objective and constraints ---
        obj = cp.Minimize(cp.pf_eigenvalue(B))

        constraints = [
            cp.prod(B[free_rows, free_cols]) == 1.0,  # product of all free entries
            B[inds[:, 0], inds[:, 1]] == a_val,      # enforce known entries
        ]

        prob = cp.Problem(obj, constraints)

        # Solve the problem – catch *only* solver errors
        try:
            result = prob.solve(gp=True, warm_start=True)
        except cp.SolverError:
            return None
        except Exception:
            return None

        # If the solver failed to find a feasible solution
        if prob.status not in ["optimal", "optimal_inaccurate"]:
            return None

        B_val = B.value
        if B_val is None:
            return None

        return {"B": B_val.tolist(), "optimal_value": result}