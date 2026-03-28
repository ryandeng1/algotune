import numpy as np
from scipy.optimize import linprog

class Solver:
    def solve(self, problem):
        """
        Solve the LP box problem using scipy's linprog (HiGHS solver).

        The problem has the form:
            minimize     c^T x
            subject to   A x <= b
                         0 <= x <= 1

        Parameters
        ----------
        problem : dict
            Dictionary containing:
                x['c']  -  list or array of objective coefficients
                x['A']  -  2D list or array of inequality matrix
                x['b']  -  list or array of inequality bounds

        Returns
        -------
        dict
            Dictionary with key 'solution' containing a list of the optimal
            variable values.
        """
        c   = np.asarray(problem['c'], dtype=float)
        A   = np.asarray(problem['A'], dtype=float)
        b   = np.asarray(problem['b'], dtype=float)

        # bounds 0 <= x_i <= 1
        bounds = [(0.0, 1.0)] * c.size

        # Solve the LP
        res = linprog(c, A_ub=A, b_ub=b, bounds=bounds,
                      method='highs', options={'presolve': True})

        # In case of strict failure, fall back to CLARABEL via CVXPY
        if not res.success:
            try:
                import cvxpy as cp
                x = cp.Variable(c.size)
                prob = cp.Problem(cp.Minimize(c.T @ x),
                                  [A @ x <= b, 0 <= x, x <= 1])
                prob.solve(solver='CLARABEL')
                assert prob.status == 'optimal'
                return {'solution': x.value.tolist()}
            except Exception:
                raise RuntimeError('Linear program infeasible or unbounded')
        return {'solution': res.x.tolist()}