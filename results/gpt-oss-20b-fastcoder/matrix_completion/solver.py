from typing import Any, Dict, List, Union
import cvxpy as cp
import numpy as np

class Solver:

    def solve(self, problem: Dict[str, Union[List[List[int]], List[float], int]]) -> Dict[str, Union[List[List[float]], float]]:
        """
        Solve the Perron–Frobenius matrix completion with CVXPY.

        Parameters
        ----------
        problem : dict
            Keys:
                - 'inds': list of index pairs (row, col) that are fixed
                - 'a'   : list of corresponding fixed values
                - 'n'   : dimension of the square matrix

        Returns
        -------
        dict
            - 'B'             : completed matrix (list of lists)
            - 'optimal_value' : optimal Perron–Frobenius eigenvalue
            (or None if solver fails)
        """
        n = problem['n']
        inds = np.array(problem['inds'], dtype=int)
        a = np.array(problem['a'], dtype=float)

        # All indices of an n×n matrix
        rows = np.arange(n)[:, None]
        cols = np.arange(n)[None, :]
        allinds = np.column_stack((rows.ravel(), cols.ravel()))

        # Find the indices that are NOT fixed (mask complement)
        fixed_mask = np.isin(allinds[:, None], inds, assume_unique=True).all(2).any(0)
        otherinds = allinds[~fixed_mask]

        # CVXPY variable and problem
        B = cp.Variable((n, n), pos=True)
        constraints = [
            cp.prod(B[otherinds[:, 0], otherinds[:, 1]]) == 1.0,
            B[inds[:, 0], inds[:, 1]] == a
        ]
        objective = cp.Minimize(cp.pf_eigenvalue(B))
        prob = cp.Problem(objective, constraints)

        try:
            result = prob.solve(gp=True)
        except (cp.SolverError, Exception):
            return None

        if B.value is None:
            return None

        return {'B': B.value.tolist(), 'optimal_value': result}