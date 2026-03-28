from typing import Any
import numpy as np
import scipy.ndimage

class Solver:
    """
    Optimised solver for 2‑D shift using scipy.ndimage.shift.
    """
    __slots__ = ("mode", "order")

    def __init__(self):
        # Basic parameters – fine for most images
        self.mode = "constant"
        self.order = 3

    def _shift_image(self, img: np.ndarray, shift: tuple[float, float]) -> np.ndarray:
        """
        Apply sub‑pixel shift to a 2‑D array.

        Parameters
        ----------
        img : np.ndarray
            The image to shift.  Converted to float32 for speed.
        shift : tuple[float, float]
            (shift_y, shift_x) in pixel units.

        Returns
        -------
        np.ndarray
            Shifted image.
        """
        if img.ndim != 2:
            raise ValueError("Only 2‑D images are supported.")
        # Ensure contiguous and float32 to reduce memory traffic
        img = np.ascontiguousarray(img, dtype=np.float32)
        return scipy.ndimage.shift(
            img, shift, order=self.order, mode=self.mode, prefilter=False
        )

    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Solve the shift problem.

        Parameters
        ----------
        problem : dict
            Must contain keys 'image' and 'shift'.

        Returns
        -------
        dict
            {'shifted_image': np.ndarray} or
            {'shifted_image': []} on failure.
        """
        img = problem.get("image")
        shift = problem.get("shift")
        if img is None or shift is None:
            return {"shifted_image": []}
        try:
            shifted = self._shift_image(img, shift)
        except Exception:
            return {"shifted_image": []}
        return {"shifted_image": shifted}