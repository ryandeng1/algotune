from typing import Dict, List, Union
import cvxpy as cp
import numpy as np

class Solver:
    def solve(
        self,
        problem: Dict[
            str,
            Union[List[List[int]], List[float], int]
        ]
    ) -> Dict[str, Union[List[List[float]], float]]:
        # Extract data
        inds = np.asarray(problem["inds"], dtype=int)
        a = np.asarray(problem["a"], dtype=float)
        n = int(problem["n"])

        # Build matrix of all indices
        allinds = np.arange(n * n).reshape(n, n)
        flat = np.arange(n * n)
        # Logical mask of known entries
        mask = np.full(n * n, False, bool)
        mask[inds[:, 0] * n + inds[:, 1]] = True
        # Indices of variables to constrain product
        other_mask = ~mask
        other_inds = np.column_stack(np.where(other_mask))

        # CVXPY variable and constraints
        B = cp.Variable((n, n), pos=True)
        constraints = [
            B[inds[:, 0], inds[:, 1]] == a,
            cp.prod(B[other_inds[:, 0], other_inds[:, 1]]) == 1.0,
        ]

        # Objective: minimize Perron root
        objective = cp.Minimize(cp.pf_eigenvalue(B))

        prob = cp.Problem(objective, constraints)
        try:
            val = prob.solve(gp=True)
        except Exception:
            return None

        if B.value is None:
            return None

        return {"B": B.value.tolist(), "optimal_value": val}