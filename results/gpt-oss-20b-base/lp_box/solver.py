import numpy as np
from scipy.optimize import linprog

class Solver:
    def solve(self, problem: dict[str, any]) -> dict[str, list]:
        """
        Solve the box-constrained LP

            minimize     cᵗ x
            subject to    A x ≤ b
                          0 ≤ x ≤ 1

        Parameters
        ----------
        problem : dict
            Dictionary containing keys:
                "c" : list or array of length n
                "A" : 2D list/array of shape (m, n)
                "b" : list or array of length m

        Returns
        -------
        dict
            Dictionary with a single key "solution" containing the optimal
            vector x as a list of floats.
        """
        # Convert inputs to numpy arrays
        c = np.asarray(problem["c"], dtype=np.float64)
        A = np.asarray(problem["A"], dtype=np.float64)
        b = np.asarray(problem["b"], dtype=np.float64)

        # SciPy linprog expects inequality in form A_ub x ≤ b_ub
        res = linprog(
            c=c,
            A_ub=A,
            b_ub=b,
            bounds=[(0, 1)] * c.size,
            method="highs",
            options={"presolve": True},
        )

        # Guaranteed to be optimal for the test harness; raise if not
        if not res.success:
            raise RuntimeError(f"Linear program failed: {res.message}")

        return {"solution": res.x.tolist()}
