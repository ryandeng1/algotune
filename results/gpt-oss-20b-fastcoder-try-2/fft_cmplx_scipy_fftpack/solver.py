import numpy as np
from numpy.typing import NDArray

class Solver:

    def solve(self, problem: NDArray) -> NDArray:
        """
        Compute the N-dimensional FFT using the fastest built‑in NumPy routine.
        """
        # Ensure a contiguous array for best performance
        data = np.asarray(problem, dtype=np.result_type(problem))
        return np.fft.fftn(data)