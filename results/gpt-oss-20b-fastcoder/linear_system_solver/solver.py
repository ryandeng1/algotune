import numpy as np
from typing import Any, List

class Solver:
    def solve(self, problem: dict[str, Any]) -> List[float]:
        """
        Solve the linear system Ax = b using NumPy's optimized solver.

        Parameters
        ----------
        problem : dict
            Dictionary containing the keys 'A' and 'b', which describe the
            linear system.  The values are expected to be array‑like.

        Returns
        -------
        list[float]
            The solution vector x.
        """
        # Use ``np.asarray`` instead of ``np.array`` to avoid an unnecessary
        # copy when the input is already a NumPy array.
        A = np.asarray(problem["A"])
        b = np.asarray(problem["b"])

        # ``np.linalg.solve`` is the fastest routine for solving square systems.
        # It returns a NumPy array; ``tolist`` converts it to a plain Python list.
        return np.linalg.solve(A, b).tolist()