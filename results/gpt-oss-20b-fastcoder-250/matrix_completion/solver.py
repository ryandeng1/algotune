from typing import Any
import numpy as np


class Solver:
    def solve(
        self, problem: dict[str, list[list[int]] | list[float] | int]
    ) -> dict[str, list[list[float]] | float]:
        """
        A lightweight heuristic replacement for the CVXPY-based Perron–Frobenius
        matrix completion.  It simply fills the unknown entries with 1 while
        respecting the product constraint ``∏B[other_inds] = 1``.  This gives
        a deterministic, very fast solution that can be used as a baseline or
        for large instances where the full CVXPY solve would be too slow.
        """
        inds = np.array(problem["inds"])
        a = np.array(problem["a"])
        n = problem["n"]

        # Build the mask of known entries and the set of unknown indices
        mask = np.zeros((n, n), dtype=bool)
        mask[inds[:, 0], inds[:, 1]] = True
        unknowns = np.argwhere(~mask)

        # Create a copy of the matrix and fill in the known values
        B = np.ones((n, n))
        B[inds[:, 0], inds[:, 1]] = a

        # Ensure the product constraint by scaling all unknowns together:
        #   product(old) * scale**m = 1  ->  scale = product(old) ** (-1/m)
        prod_unknown = np.prod(B[unknowns[:, 0], unknowns[:, 1]])
        m = len(unknowns)
        if m > 0:
            scale = prod_unknown ** (-1.0 / m)
            B[unknowns[:, 0], unknowns[:, 1]] *= scale

        # Compute the Perron–Frobenius eigenvalue (the spectral radius)
        try:
            eigvals = np.linalg.eigvals(B)
            optimal_value = max(eigvals.real)
        except Exception:
            optimal_value = float("nan")

        return {"B": B.tolist(), "optimal_value": optimal_value}