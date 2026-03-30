import numpy as np
from numpy.linalg import solve as _solve

class Solver:
    """
    High‑performance linear system solver.

    The class converts the input matrices to NumPy arrays once,
    uses the highly optimised BLAS/LAPACK routines in `np.linalg.solve`,
    and returns the solution in the required list format.
    """
    def __init__(self) -> None:
        # No heavy initialisation; the import above is sufficient.
        pass

    def solve(self, problem: dict[str, object]) -> list[float]:
        """
        Solve a linear system Ax = b.

        Parameters
        ----------
        problem : dict
            Dictionary with keys 'A' and 'b'. The values can be any
            array‑like objects that NumPy can convert to array.

        Returns
        -------
        list[float]
            Solution vector `x` converted to a plain Python list.
        """
        # Convert to NumPy arrays without copying if possible.
        # `dtype=np.float64` guarantees the same precision as ndarray default.
        A = np.asarray(problem['A'], dtype=np.float64)
        b = np.asarray(problem['b'], dtype=np.float64)

        # Solve using the fast LAPACK routine.
        x = _solve(A, b)

        # Convert NumPy array to list (Python primitives).
        return x.tolist()