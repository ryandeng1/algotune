# solver.py
import numpy as np
from numpy.typing import NDArray

class Solver:
    """
    Solver that computes the N‑dimensional Fast Fourier Transform of a problem
    array using the highly tuned `numpy.fft.fftn` implementation.
    """

    def solve(self, problem: NDArray) -> NDArray:
        """
        Compute the N-dimensional FFT of `problem`.

        Parameters
        ----------
        problem : NDArray
            Multi‑dimensional input array.

        Returns
        -------
        NDArray
            The FFT of the input array.
        """
        return np.fft.fftn(problem)