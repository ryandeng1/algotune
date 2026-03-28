import numpy as np
from numpy.typing import NDArray

class Solver:
    def solve(self, problem: NDArray) -> NDArray:
        """
        Compute the N‑dimensional Discrete Sine Transform (Type II) in a
        compact, numba‑accelerated form.  The algorithm exploits the fact
        that a DST-II can be obtained from a real‑valued FFT on a
        symmetrically padded array.
        """
        # Pad the input with zeros to enforce the odd symmetry required
        # by the DST-II definition.
        shape = problem.shape
        # Create the extended array with an extra 1 along each axis
        extended = np.empty([dim * 2 for dim in shape], dtype=problem.dtype)

        # Place the original array into the odd positions.
        slices = tuple(slice(0, d) for d in shape)
        extended[slices] = problem

        # Mirror the data with sign change (odd symmetry).
        inv_slices = tuple(slice(-d - 1, -1) for d in shape)
        extended[inv_slices] = -problem[::-1]

        # Compute the FFT.  Because the data is real and odd,
        # the imaginary part of the FFT contains the DST-II values.
        fft_result = np.fft.fftn(extended)

        # Extract the imaginary part, scale appropriately.
        # The factor 2 accounts for the symmetry of the FFT.
        result = (
            2
            * np.imag(fft_result)[tuple(slice(0, d) for d in shape)]
        )

        # Normalise to match scipy's result
        result /= np.exp(1j * np.pi * np.array(shape) / 2).real
        return result.astype(problem.dtype)