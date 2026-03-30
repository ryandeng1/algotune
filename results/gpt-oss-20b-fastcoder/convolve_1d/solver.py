from typing import Tuple
import numpy as np

class Solver:
    """
    A minimal solver that performs convolution of two 1‑D signals.
    The implementation delegates to NumPy's highly optimised C routine
    instead of the SciPy wrapper, which reduces the Python overhead.
    """
    __slots__ = ("mode",)

    def __init__(self) -> None:
        # Using `full` is the default behaviour in the original code.
        self.mode: str = "full"

    def solve(self, problem: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        a, b = problem
        # NumPy's np.convolve is a thin wrapper around the same
        # underlying C routine used by SciPy, but it involves
        # fewer Python calls and therefore runs slightly faster.
        return np.convolve(a, b, mode=self.mode)