import numpy as np
import cvxpy as cp

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any] | None:
        """
        Solves the Perron‑Frobenius matrix completion.

        Parameters
        ----------
        problem
            Dictionary with keys:
            - 'n'   : matrix size
            - 'inds': shape (k,2) indices with fixed entries
            - 'a'   : shape (k,) known values at ``inds``

        Returns
        -------
        dict or None
            Dictionary containing the optimal matrix ``B`` and its
            Perron‑Frobenius eigenvalue, or ``None`` on failure.
        """
        n = int(problem["n"])
        inds = np.asarray(problem["inds"], dtype=np.int64)
        a = np.asarray(problem["a"], dtype=np.float64)

        # All index pairs in column-major order as two 1‑D arrays
        rows = np.repeat(np.arange(n), n)
        cols = np.tile(np.arange(n), n)

        # Create a mask of indices that are *not* in the fixed set
        mask = np.ones(n * n, dtype=bool)
        for r, c in inds:
            mask[r + c * n] = False  # column-major indexing

        # Prepare variable and constraints
        B = cp.Variable((n, n), pos=True)
        obj = cp.Minimize(cp.pf_eigenvalue(B))

        con = [
            # Product of unconstrained entries must equal 1
            cp.prod(B[rows[mask], cols[mask]]) == 1.0,
            # Fix the known entries
            B[inds[:, 0], inds[:, 1]] == a
        ]

        prob = cp.Problem(obj, con)
        try:
            val = prob.solve(gp=True, verbose=False, max_iters=2000)
        except (cp.SolverError, RuntimeError):
            return None

        if B.value is None:
            return None
        return {"B": B.value.tolist(), "optimal_value": float(val)}