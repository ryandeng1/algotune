import numpy as np

class Solver:
    """
    Fast matrix multiplication solver.
    """
    def solve(self, problem: dict[str, list[list[float]]]) -> list[list[float]]:
        """
        Compute the product C = A · B for two given matrices.

        Parameters
        ----------
        problem : dict
            Dictionary with keys 'A' and 'B', each mapping to a nested list of floats.

        Returns
        -------
        list[list[float]]
            The product matrix as a nested list.
        """
        # Convert to NumPy arrays once (the fastest way to build a contiguous matrix)
        A = np.array(problem['A'], dtype=np.float64, copy=False)
        B = np.array(problem['B'], dtype=np.float64, copy=False)

        # NumPy's matmul (or dot) is heavily optimized; using the `@` operator is
        # slightly faster as it dispatches to the best implementation.
        C = A @ B

        # Convert back to plain Python nested lists for the required output format.
        return C.tolist()