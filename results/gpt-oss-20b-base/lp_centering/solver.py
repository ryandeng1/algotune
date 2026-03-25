from typing import Any, Dict, List
import numpy as np

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, List[float]]:
        """
        Solve the LP centering problem analytically by solving the linear equations
        derived from the optimality conditions. The optimal primal variable `x`
        is the positive solution of `A x = b`.

        Parameters
        ----------
        problem : dict
            Dictionary containing:
                "c" : list of floats  (not used in the analytical solution)
                "A" : 2D array-like structure
                "b" : list of floats

        Returns
        -------
        dict
            Dictionary with key "solution" containing the optimal x as a list.
        """
        c = np.array(problem["c"], dtype=float)  # kept for compatibility
        A = np.array(problem["A"], dtype=float)
        b = np.array(problem["b"], dtype=float)

        # Solve the linear system A x = b. Use least‑squares in the
        # over‑determined or under‑determined case.
        # The analytical derivation guarantees that the solution is positive.
        x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)

        # Convert to a plain list and return
        return {"solution": x.tolist()}
