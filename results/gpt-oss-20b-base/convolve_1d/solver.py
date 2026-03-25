import numpy as np


class Solver:
    """Inverse‑transform 1‑D correlation solver."""

    def __init__(self, mode: str = "full"):
        """
        Parameters
        ----------
        mode : str, optional
            Convolution mode: 'full', 'valid' or 'same'. Defaults to 'full'.

        The constructor runs only once per solver instance and does not count
        towards the per‑call runtime budget.
        """
        if mode not in {"full", "valid", "same"}:
            raise ValueError(f"Unsupported mode: {mode!r}")
        self.mode = mode

    def solve(self, problem, **kwargs):
        """
        Compute the one‑dimensional correlation of each pair in *problem*.

        Parameters
        ----------
        problem : list[tuple[np.ndarray, np.ndarray]]
            A list of pairs of one‑dimensional arrays to convolve.

        Returns
        -------
        list[np.ndarray]
            The list of correlation results, one per input pair.
        """
        out = []

        # Local alias for speed
        mode = self.mode

        for a, b in problem:
            # Ensure inputs are numpy arrays of dtype float64 for consistency
            a = np.asarray(a, dtype=np.float64).ravel()
            b = np.asarray(b, dtype=np.float64).ravel()

            # Delegate to numpy.convolve which is highly optimised in C
            out.append(np.convolve(a, b, mode=mode))

        return out
