import numpy as np

class Solver:
    def solve(self, problem, **kwargs) -> np.ndarray:
        """
        Compute the full 1-D convolution of two input arrays.

        Parameters
        ----------
        problem : tuple
            A tuple (a, b) where `a` and `b` are 1-D sequences of numbers
            (lists, tuples, or numpy arrays).  They may be of different
            lengths.

        Returns
        -------
        np.ndarray
            The full convolution of `a` and `b` as a 1-D numpy array.
        """
        a, b = problem
        # Convert inputs to 1-D float arrays; this handles lists/tuples
        a_arr = np.asarray(a, dtype=np.float64).ravel()
        b_arr = np.asarray(b, dtype=np.float64).ravel()
        # Use numpy's fast one‑dimensional convolution
        return np.convolve(a_arr, b_arr, mode="full")
