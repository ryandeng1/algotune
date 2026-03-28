import numpy as np

class Solver:
    """
    Optimised 2‑D correlation for dense arrays.
    Uses FFT‑based convolution with linear padding to achieve
    the same result as scipy.signal.correlate2d with
    mode = 'full', boundary = 'fill'.
    """

    def _correlate_full_fft(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Return full 2‑D correlation of a and b using FFT.
        Parameters
        ----------
        a, b : np.ndarray
            2‑D input arrays (float‑like).

        Returns
        -------
        result : np.ndarray
            Correlation array of shape
            (a.shape[0] + b.shape[0] - 1,
             a.shape[1] + b.shape[1] - 1).
        """
        # Flip kernel for correlation
        k = np.flipud(np.fliplr(b))

        # Sizes for linear convolution
        shape_a = a.shape
        shape_k = k.shape
        out_shape = (shape_a[0] + shape_k[0] - 1,
                     shape_a[1] + shape_k[1] - 1)

        # Pad arrays to output size
        fft_shape = [1 << (n-1).bit_length() for n in out_shape]
        A = np.fft.rfftn(a, s=fft_shape)
        K = np.fft.rfftn(k, s=fft_shape)

        # Element‑wise multiplication and inverse FFT
        C = A * K
        corr = np.fft.irfftn(C, s=fft_shape)

        # Trim to exact output size
        return corr[:out_shape[0], :out_shape[1]]

    def solve(self, problem: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """
        Compute the 2D correlation of arrays a and b using "full" mode
        and "fill" boundary by applying FFT‑based convolution.

        Parameters
        ----------
        problem : tuple
            A tuple (a, b) of 2D arrays.

        Returns
        -------
        np.ndarray
            Correlation result.
        """
        a, b = problem
        # Ensure float dtype for FFT
        if not np.issubdtype(a.dtype, np.floating):
            a = a.astype(np.float64, copy=False)
        if not np.issubdtype(b.dtype, np.floating):
            b = b.astype(np.float64, copy=False)

        return self._correlate_full_fft(a, b)