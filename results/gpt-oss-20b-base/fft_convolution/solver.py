import numpy as np

class Solver:
    """
    Fast convolution solver using real FFT (rfft) for real signals.
    """

    @staticmethod
    def _convolve_real(a: np.ndarray, b: np.ndarray, mode: str = "full") -> np.ndarray:
        """
        Convolve two real numpy arrays using a single real FFT.

        Parameters
        ----------
        a : np.ndarray
            First input array.
        b : np.ndarray
            Second input array.
        mode : str, optional
            One of {'full', 'valid', 'same'}.

        Returns
        -------
        np.ndarray
            Convolution of a and b in the requested mode.
        """
        if a.ndim != 1 or b.ndim != 1:
            raise ValueError("Only 1‑D real signals are supported.")

        la, lb = a.size, b.size
        lres = la + lb - 1
        n = np.fft.next_fast_len(lres)

        # Compute FFTs
        fa = np.fft.rfft(a, n=n)
        fb = np.fft.rfft(b, n=n)

        # Element‑wise multiplication in frequency domain
        fc = fa * fb

        # Inverse FFT to data domain
        conv_full = np.fft.irfft(fc, n=n)[:lres]

        # Downsample to requested mode
        if mode == "full":
            return conv_full
        elif mode == "same":
            start = (lb - 1) // 2
            return conv_full[start : start + la]
        elif mode == "valid":
            return conv_full[(lb - 1) : -(la - 1) if lb > 1 else None]
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def solve(self, problem: dict[str, any]) -> dict[str, any]:
        """
        Compute the convolution of two sequences using real FFT.

        Parameters
        ----------
        problem : dict
            Dictionary containing:
                - 'signal_x': Iterable of real numbers.
                - 'signal_y': Iterable of real numbers.
                - 'mode': One of 'full', 'same', 'valid' (default: 'full').

        Returns
        -------
        dict
            A dictionary with the key 'convolution' mapping to the result list.
        """
        a = np.asarray(problem["signal_x"], dtype=float)
        b = np.asarray(problem["signal_y"], dtype=float)
        mode = problem.get("mode", "full")

        conv = self._convolve_real(a, b, mode=mode)
        return {"convolution": conv.tolist()}