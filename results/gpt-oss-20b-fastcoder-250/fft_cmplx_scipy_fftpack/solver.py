import numpy as np
from numpy.typing import NDArray
from typing import Any

class Solver:
    def solve(self, problem: NDArray, **kwargs) -> Any:
        """
        Compute the N-dimensional Fast Fourier Transform of the input complex array.

        Parameters
        ----------
        problem : NDArray
            Input n×n complex array.

        Returns
        -------
        NDArray
            N-dimensional FFT of the input.
        """
        # Use numpy's highly optimized FFT implementation
        return np.fft.fftn(problem)
