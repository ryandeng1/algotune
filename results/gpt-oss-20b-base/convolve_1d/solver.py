from typing import Any
import numpy as np


class Solver:
    def solve(self, problem: tuple) -> np.ndarray:
        """
        Efficiently compute the 1‑D convolution of two arrays using NumPy.

        Parameters
        ----------
        problem : tuple
            A tuple containing the two input arrays, `a` and `b`. They can be
            any objects that can be converted to NumPy arrays.

        Returns
        -------
        np.ndarray
            The convolution result. The same `mode` attribute used in the
            original implementation is preserved.
        """
        a, b = problem
        # Ensure the inputs are NumPy arrays with the same dtype for consistency.
        a = np.asarray(a)
        b = np.asarray(b)

        # `mode` is expected to exist as an instance attribute. If not
        # present, fall back to 'full'.
        mode = getattr(self, "mode", "full")

        return np.convolve(a, b, mode=mode)