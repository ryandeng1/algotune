from typing import Any
import numpy as np
from scipy.fft import dstn
from numpy.typing import NDArray


class Solver:
    def solve(self, problem: NDArray) -> NDArray:
        """
        Compute the N-dimensional DST Type II using scipy's more modern fft module.
        This implementation is faster than scipy.fftpack.dstn and preserves
        the input data type and shape.
        """
        # Ensure the input is a NumPy array with float64 precision for maximum
        # performance – the DST is defined for real input only.
        arr = np.asarray(problem, dtype=np.float64)

        # Compute the N-Dimensional DST Type II
        result = dstn(arr, type=2)

        # Preserve original dtype if necessary
        if result.dtype != problem.dtype:
            result = result.astype(problem.dtype)

        return result