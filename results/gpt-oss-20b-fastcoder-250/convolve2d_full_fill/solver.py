import numpy as np

class Solver:
    """Fast 2‑D full convolution using the FFT (real‑valued)."""

    @staticmethod
    def _fft2d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Compute the full 2‑D convolution of real arrays a and b using the FFT.

        Parameters
        ----------
        a : np.ndarray
            Input array of shape (H1, W1).
        b : np.ndarray
            Input array of shape (H2, W2).

        Returns
        -------
        np.ndarray
            Full convolution result of shape (H1 + H2 - 1, W1 + W2 - 1).
        """
        H1, W1 = a.shape
        H2, W2 = b.shape

        out_h = H1 + H2 - 1
        out_w = W1 + W2 - 1

        # Compute the size for the FFT: next power of 2 for speed (optional)
        fft_h = 1 << (out_h - 1).bit_length()
        fft_w = 1 << (out_w - 1).bit_length()

        # Zero‑pad inputs to the FFT size
        fft_a = np.fft.rfftn(a, s=(fft_h, fft_w))
        fft_b = np.fft.rfftn(b, s=(fft_h, fft_w))

        # Element‑wise multiplication (convolution theorem)
        conv_fft = fft_a * fft_b

        # Inverse FFT to get spatial domain; only real part is needed
        conv_full = np.fft.irfftn(conv_fft, s=(fft_h, fft_w))

        # Crop to the exact full size
        return conv_full[:out_h, :out_w]

    def solve(self, problem: tuple, **kwargs) -> np.ndarray:
        """
        Compute the full 2‑D convolution of two real 2‑D arrays using FFT.

        Parameters
        ----------
        problem : tuple
            A tuple (a, b) where a and b are 2‑D numpy arrays.
        **kwargs : dict
            Additional keyword arguments are ignored.

        Returns
        -------
        np.ndarray
            The full convolution result.
        """
        a, b = problem
        # Ensure inputs are 2‑dimensional contiguous arrays
        a = np.ascontiguousarray(a, dtype=np.float64)
        b = np.ascontiguousarray(b, dtype=np.float64)
        return self._fft2d(a, b)
