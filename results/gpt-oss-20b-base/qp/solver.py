# solver.py
import numpy as np
from typing import Any, Dict
from scipy.optimize import minimize

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Solve a convex quadratic program:

            min 1/2 x^T Q x + q^T x
            s.t. G x <= h
                 A x == b

        Parameters
        ----------
        problem : dict
            Dictionary with keys:
                "Q" : (n, n) array
                "q" : (n,) array
                "G" : (m, n) array
                "h" : (m,) array
                "A" : (p, n) array
                "b" : (p,) array

        Returns
        -------
        dict
            {"solution": list of n floats, "objective": float}
        """
        # convert to numpy arrays
        Q = np.asarray(problem["Q"], dtype=float)
        q = np.asarray(problem["q"], dtype=float)
        G = np.asarray(problem["G"], dtype=float)
        h = np.asarray(problem["h"], dtype=float)
        A = np.asarray(problem["A"], dtype=float)
        b = np.asarray(problem["b"], dtype=float)

        n = Q.shape[0]

        # objective function and gradient
        def f(x):
            return 0.5 * np.dot(x, Q @ x) + np.dot(q, x)

        def grad(x):
            return Q @ x + q

        # Linear constraints
        cons = []
        if G.shape[0] > 0:
            # inequality: Gx <= h ->  Gx - h <= 0
            cons.append(
                {"type": "ineq",
                 "fun": lambda x: h - G @ x,
                 "jac": lambda x: -G}
            )
        if A.shape[0] > 0:
            cons.append(
                {"type": "eq",
                 "fun": lambda x: A @ x - b,
                 "jac": lambda x: A}
            )

        # initial guess
        x0 = np.zeros(n)

        # call solver
        res = minimize(
            fun=f,
            x0=x0,
            jac=grad,
            constraints=cons,
            method="trust-constr",
            options={"gtol": 1e-9, "xtol": 1e-9, "maxiter": 1000, "verbose": 0}
        )

        if not res.success:
            raise ValueError(f"Solver failed: {res.message}")

        solution = res.x
        objective = f(solution)

        return {"solution": solution.tolist(), "objective": float(objective)}
