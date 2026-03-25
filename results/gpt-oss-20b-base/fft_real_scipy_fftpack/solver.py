import numpy as np
from numpy.typing import NDArray

class Solver:
    def solve(self, problem: NDArray, **kwargs) -> NDArray:
        """
        Compute the N-dimensional Fast Fourier Transform of a real-valued array.

        The implementation uses :func:`numpy.fft.fftn`, which is highly optimized
        and offers performance comparable to scipy's implementation without the
        overhead of importing scipy.  It directly returns the complex spectrum
        expected by the evaluation harness.
        """
        # Directly use numpy's FFT; it handles arbitrary ND arrays and real inputs gracefully.
        return np.fft.fftn(problem)
