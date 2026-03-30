# solver.py
import numpy as np
from scipy import signal
from numba import njit, jit, prange

# ----------------------------------------------------------------------
# Helper to perform convolution via FFT in a fast, explicit Numba implementation.
# ----------------------------------------------------------------------
@njit
def _convolve_fft(x: np.ndarray, y: np.ndarray, mode: str = 'full') -> np.ndarray:
    """
    Convolve two real 1-D signals using an FFT based algorithm.
    Works for 'full', 'valid', and 'same' modes.

    Parameters
    ----------
    x, y : np.ndarray
        Real 1-D input signals.
    mode : str
        The convolution mode. Must be one of 'full', 'same', 'valid'.

    Returns
    -------
    np.ndarray
        The convolution result.
    """
    # Determine the length for zero-padded FFTs
    n_x = x.shape[0]
    n_y = y.shape[0]
    n_full = n_x + n_y - 1

    # Choose the next power of two for efficient FFT
    n_fft = 1 << (n_full - 1).bit_length()

    # Zero-pad inputs
    X = np.empty(n_fft, dtype=np.complex128)
    Y = np.empty(n_fft, dtype=np.complex128)
    X.fill(0.0)
    Y.fill(0.0)
    X[:n_x] = x
    Y[:n_y] = y

    # Forward FFTs
    X_fft = np.fft.rfft(X, n_fft)
    Y_fft = np.fft.rfft(Y, n_fft)

    # Pointwise multiplication
    conv_fft = X_fft * Y_fft

    # Inverse FFT
    conv = np.fft.irfft(conv_fft, n_fft)

    # Truncate to the requested mode
    if mode == 'full':
        return conv[:n_full]
    elif mode == 'same':
        start = (n_y - 1) // 2
        end = start + n_x
        return conv[start:end]
    elif mode == 'valid':
        return conv[n_y - 1 : n_x]
    else:
        raise ValueError(f"Unsupported mode: {mode}")


# ----------------------------------------------------------------------
# Main solver class
# ----------------------------------------------------------------------
class Solver:
    """
    Solver for 1-D convolution problems using an efficient FFT approach.
    """

    @staticmethod
    def solve(problem: dict) -> dict[str, list]:
        """
        Compute the convolution of two signals using a fast FFT implementation.

        Parameters
        ----------
        problem : dict
            Dictionary containing:
                - 'signal_x': Iterable of real numbers.
                - 'signal_y': Iterable of real numbers.
                - Optional 'mode': One of 'full', 'same', or 'valid'.

        Returns
        -------
        dict
            Dictionary with a single key 'convolution' mapping to the
            convolution result as a Python list.
        """
        # Convert inputs to 1-D float array
        x = np.asarray(problem['signal_x'], dtype=np.float64).ravel()
        y = np.asarray(problem['signal_y'], dtype=np.float64).ravel()

        # Default mode is 'full' (same as scipy.signal.fftconvolve)
        mode = problem.get('mode', 'full')

        # Validate mode
        if mode not in ('full', 'same', 'valid'):
            raise ValueError(f"Unsupported mode: {mode}")

        # Fast FFT-based convolution (numba compiled)
        conv = _convolve_fft(x, y, mode)

        return {"convolution": conv.tolist()}