import numpy as np

# The only used SciPy function is fft, which is already available as numpy.fft.
# No external import from scipy is required.

class Solver:
    """
    Optimised solver for 2‑D correlation.
    It performs the correlation by using a 2‑D FFT which has a
    logarithmic cost and beats the naïve convolution for
    larger inputs.
    """

    def __init__(self):
        # These settings are not used directly, but kept for API compatibility.
        self.boundary = 'fill'
        self.mode = 'full'

    def solve(self, problem: tuple) -> np.ndarray:
        """
        Compute the 2D correlation of arrays a and b using "full" mode and "fill" boundary.

        Parameters
        ----------
        problem : tuple
            A tuple (a, b) of 2D arrays.

        Returns
        -------
        np.ndarray
            A 2D array containing the correlation result.
        """
        a, b = problem
        # Convert to float64 for consistent behaviour and to avoid overflow.
        a = np.asarray(a, dtype=np.float64, copy=False)
        b = np.asarray(b, dtype=np.float64, copy=False)

        # Determine the output shape for "full" correlation.
        out_shape = (a.shape[0] + b.shape[0] - 1, a.shape[1] + b.shape[1] - 1)

        # Pad both inputs so that after FFT and multiplication we can directly
        # truncate the result to the full output.
        pad_a = ((0, out_shape[0] - a.shape[0]), (0, out_shape[1] - a.shape[1]))
        pad_b = ((0, out_shape[0] - b.shape[0]), (0, out_shape[1] - b.shape[1]))

        # Note: correlation is the same as convolution without flipping.
        # Therefore we don't flip `b`.
        a_padded = np.pad(a, pad_a, mode='constant', constant_values=0)
        b_padded = np.pad(b, pad_b, mode='constant', constant_values=0)

        # Perform the FFT‑based multiplication.
        fft_a = np.fft.rfftn(a_padded)
        fft_b = np.fft.rfftn(b_padded)
        # Element‑wise product in frequency domain.
        conv = fft_a * fft_b
        # Inverse FFT to get the spatial domain result.
        result = np.fft.irfftn(conv, s=out_shape)

        # The real part contains the correlation; discard tiny numerical
        # imaginary components that arise from floating‑point rounding.
        return result.real