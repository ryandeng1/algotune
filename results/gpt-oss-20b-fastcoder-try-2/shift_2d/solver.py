import numpy as np
import scipy.ndimage

class Solver:
    """
    A lightweight solver that applies a 2‑D shift to an image.
    Implements the public interface expected by the evaluation harness
    while avoiding unnecessary Python overhead.
    """
    def __init__(self) -> None:
        # Parameters are still kept for configurability but are applied
        # directly via the wrapped scipy function.
        self.mode = 'constant'
        self.order = 3

    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Shift the image by the provided offset.

        Parameters
        ----------
        problem : dict
            Must contain keys:
            - ``image`` : 2‑D NumPy array
            - ``shift`` : tuple of two floats (dy, dx)

        Returns
        -------
        dict
            ``{'shifted_image': np.ndarray}``
        """
        image = problem["image"]
        shift_vector = problem["shift"]

        # ``scipy.ndimage.shift`` is highly optimised (Cython + SIMD);
        # the call below is the fastest generic implementation in CPython.
        shifted_image = scipy.ndimage.shift(
            image,
            shift=shift_vector,
            order=self.order,
            mode=self.mode,
            prefilter=True,
        )

        return {"shifted_image": shifted_image}