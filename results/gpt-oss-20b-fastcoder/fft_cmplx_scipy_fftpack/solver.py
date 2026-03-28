from typing import Any
import numpy as np
from numpy.typing import NDArray

class Solver:

    def solve(self, problem: NDArray) -> NDArray:
        """
        Compute the N-dimensional FFT using NumPy's FFT implementation,
        which is typically faster and more feature‑rich than scipy.fftpack.
        """
        # Ensure the input is a NumPy array and use float64 for consistency
        arr = np.asarray(problem, dtype=np.complex128)
        return np.fft.fftn(arr)