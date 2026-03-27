import numpy as np
from typing import Any


class Solver:
    def __init__(self, mode: str = "full") -> None:
        self.mode = mode

    def solve(self, problem: tuple) -> np.ndarray:
        """
        Perform a linear convolution of the two 1-D numpy arrays `a` and `b`.

        Parameters
        ----------
        problem : tuple
            A pair of 1‑D numeric sequences (a, b) to be convolved.

        Returns
        -------
        np.ndarray
            The convolution result with the selected `mode`:
                * 'full'  –  the full discrete linear convolution.
                * 'same'  –  the central part of the convolution.
                * 'valid' –  the intersection of the input sequences.
        """
        a, b = problem
        # Convolve using NumPy's fast implementation.
        conv = np.convolve(a, b, mode=self.mode)
        return conv