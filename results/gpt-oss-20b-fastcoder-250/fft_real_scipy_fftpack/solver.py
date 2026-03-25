import numpy as np
from numpy.typing import NDArray
from typing import Any

class Solver:
    def solve(self, problem: NDArray, **kwargs) -> Any:
        """
        Compute the N-dimensional FFT of a real-valued matrix.

        Parameters
        ----------
        problem : NDArray
            Input n×n real-valued array.

        Returns
        -------
        NDArray
            Complex-valued array of the same shape as `problem`,
            containing the FFT of the input.
        """
        return np.fft.fftn(problem)
