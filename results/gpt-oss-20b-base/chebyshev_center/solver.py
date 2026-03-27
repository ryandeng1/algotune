from __future__ import annotations

import numpy as np
from scipy.optimize import linprog
from typing import Any, Dict, List


class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, List]:
        """
        Solve the Chebyshev center problem using SciPy's linear programming
        solver, which is typically faster than CVXPY for this particular
        LP.

        The Chebyshev centre of the polyhedron {x | a x <= b} is found by
        maximizing the radius r of the largest hyper‑ball that fits inside
        the polyhedron.  The constraints can be written linearly as

            a_i · x + r * ||a_i||_1 <= b_i   for all i.

        We treat r as an additional variable and solve the following LP:
            maximize   r
            subject to a x + r * |a| <= b
            where |a| is the element‑wise absolute value of each row
            of the matrix `a` (the 1‑norm of the row).

        Parameters
        ----------
        problem : dict
            Dictionary containing:
            - 'a': 2‑D list or numpy array of shape (m, n)
            - 'b': 1‑D list or array of length m

        Returns
        -------
        dict
            Dictionary with a single key "solution" containing a list of
            the n Chebyshev centre coordinates.
        """
        a = np.asarray(problem["a"], dtype=float)
        b = np.asarray(problem["b"], dtype=float)

        m, n = a.shape

        # Construct the constraint matrix G and vector h for
        #     G @ [x; r] <= h
        # G has shape (m, n+1)
        G = np.hstack((a, np.abs(a).sum(axis=1, keepdims=True)))
        h = b

        # Objective: maximize r  => minimize -r
        c = np.zeros(n + 1)
        c[-1] = -1.0

        # Bounds: x free (-inf, inf), r >= 0
        bounds = [(None, None)] * n + [(0, None)]

        result = linprog(c, A_ub=G, b_ub=h, bounds=bounds, method="highs")

        if not result.success:
            raise ValueError(f"Linear program failed to find optimal solution: {result.message}")

        center = result.x[:-1]  # discard radius
        return {"solution": center.tolist()}