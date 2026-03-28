from numpy.typing import NDArray
import numpy as np

class Solver:
    """
    Fast N‑dimensional FFT using NumPy's FFT implementation.
    """

    def solve(self, problem: NDArray) -> NDArray:
        """
        Compute the N‑dimensional FFT of *problem*.
        """
        # Ensure the input is a NumPy array (supports typing, also handles list/tuple)
        arr = np.asarray(problem, dtype=np.complex128)
        return np.fft.fftn(arr)