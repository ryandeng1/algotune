import numpy as np
import cvxpy as cp

class Solver:
    def solve(self, problem):
        # preprocess inputs
        inds = np.array(problem["inds"])
        a = np.array(problem["a"])
        n = problem["n"]

        # Create matrix variable
        B = cp.Variable((n, n), pos=True)

        # Objective: minimize Perron–Frobenius eigenvalue
        obj = cp.Minimize(cp.pf_eigenvalue(B))

        # Constraints
        row, col = np.indices((n, n))
        all_inds = np.column_stack([row.ravel(), col.ravel()])
        mask = np.ones(all_inds.shape[0], dtype=bool)
        for (r, c) in inds:
            mask &= ~(all_inds[:, 0] == r) | ~(all_inds[:, 1] == c)
        prod_inds = all_inds[mask]
        constraints = [
            B[inds[:, 0], inds[:, 1]] == a,
            cp.prod(B[prod_inds[:, 0], prod_inds[:, 1]]) == 1
        ]

        prob = cp.Problem(obj, constraints)
        try:
            result = prob.solve(gp=True)
        except cp.SolverError:
            return None
        except Exception:
            return None

        if B.value is None:
            return None
        return {"B": B.value.tolist(), "optimal_value": result}