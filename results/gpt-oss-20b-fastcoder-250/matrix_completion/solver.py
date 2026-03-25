import numpy as np
import scipy.linalg

class Solver:
    def solve(self, problem, **kwargs):
        # Extract data
        inds = np.array(problem["inds"])
        a = np.array(problem["a"])
        n = problem["n"]

        # Create complete matrix filled with 1 for unknowns
        B = np.ones((n, n), dtype=float)

        # Assign known values
        for (i, j), val in zip(inds, a):
            B[i, j] = val

        # Compute the Perron-Frobenius eigenvalue (dominant eigenvalue)
        # For a nonnegative matrix we can use scipy.linalg.eigvals
        eig_vals = scipy.linalg.eigvals(B, left=False, right=False)
        pf_eig = max(eig_vals.real)

        return {"B": B.tolist(), "optimal_value": float(pf_eig)}
