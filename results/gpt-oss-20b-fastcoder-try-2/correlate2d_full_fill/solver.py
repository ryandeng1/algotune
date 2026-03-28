import numpy as np

class Solver:
    __slots__ = ()

    def solve(self, problem):
        """
        Compute the 2D correlation of arrays a and b using FFT-based
        correlation in full mode. This implementation is faster than
        scipy.signal.correlate2d for large arrays because it uses a
        single FFT of the padded arrays.

        Parameters
        ----------
        problem : tuple
            A tuple containing two 2‑D NumPy arrays (a, b).

        Returns
        -------
        result : np.ndarray
            2‑D array of the correlation result.
        """
        a, b = problem

        # Check dtype of both inputs and cast to float64 if necessary
        a = a.astype(np.float64, copy=False)
        b = b.astype(np.float64, copy=False)

        # Flip kernel for correlation
        b_flip = np.flip(np.flip(b, axis=0), axis=1)

        # Compute size of the full convolution
        out_rows = a.shape[0] + b_flip.shape[0] - 1
        out_cols = a.shape[1] + b_flip.shape[1] - 1

        # Pad both arrays to the output size (next power of two for speed)
        pad_rows = 1 << (out_rows - 1).bit_length()
        pad_cols = 1 << (out_cols - 1).bit_length()

        A = np.pad(a, ((0, pad_rows - a.shape[0]), (0, pad_cols - a.shape[1])), 'constant')
        B = np.pad(b_flip, ((0, pad_rows - b_flip.shape[0]), (0, pad_cols - b_flip.shape[1])), 'constant')

        # Perform FFT on both padded arrays
        FA = np.fft.rfftn(A)
        FB = np.fft.rfftn(B)

        # Element‑wise multiplication and inverse FFT
        conv = np.fft.irfftn(FA * FB, s=(pad_rows, pad_cols))

        # Slice to the exact full output size
        result = conv[:out_rows, :out_cols]

        return result