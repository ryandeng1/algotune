import numpy as np
from scipy.optimize import linprog

class Solver:
    def solve(self, problem: dict[str, object]) -> dict[str, list]:
        """
        Solve the Chebyshev center problem using a linear program solver.

        :param problem: A dictionary containing the matrices 'a' and vector 'b'.
        :return: A dictionary with key "solution" containing the optimal point.
        """
        a = np.asarray(problem['a'], dtype=float)
        b = np.asarray(problem['b'], dtype=float).flatten()

        n = a.shape[1]                       # dimension of x
        m = a.shape[0]                       # number of constraints

        # Objective: maximise r  -> minimise -r
        # Decision variables: [x1, ..., xn, r]
        c = np.zeros(n + 1)
        c[-1] = -1.0                         # minimise -r

        # Build inequality matrix: a @ x + ||a_i|| * r <= b
        norm_a = np.linalg.norm(a, axis=1)   # shape (m,)
        A_ub = np.column_stack((a, norm_a.reshape(-1, 1)))  # shape (m, n+1)
        b_ub = b

        # Only r is unbounded; x are unconstrained (can be any real values)
        bounds = [(None, None)] * n + [(None, None)]

        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                      method='highs', options={'presolve': True})

        if not res.success:
            raise ValueError("Linear program did not converge to an optimal solution")

        x_opt = res.x[:n]
        return {"solution": x_opt.tolist()}