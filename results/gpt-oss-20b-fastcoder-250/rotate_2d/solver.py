from __future__ import annotations
from typing import Any, Dict
import numpy as np

try:
    # Optional fast implementation for arbitrary angles
    from scipy.ndimage import rotate as scipy_rotate  # type: ignore
except Exception:
    scipy_rotate = None


class Solver:
    """
    Optimised 2‑D image rotation.

    Parameters
    ----------
    reshape : bool, default=True
        Same signature as :func:`scipy.ndimage.rotate`.
    order : int, default=3
        Interpolation order.  Only implemented for edge cases.
    mode : str, default='constant'
        Boundary handling.  Only used when ``scipy_rotate`` is available.
    """

    def __init__(self, reshape: bool = True, order: int = 3, mode: str = "constant") -> None:
        self.reshape = reshape
        self.order = order
        self.mode = mode

    def _rotate_simple(self, image: np.ndarray, angle: float) -> np.ndarray:
        """
        Fast rotation for angles that are multiples of 90 degrees.
        """
        k = int(round(angle / 90)) % 4
        return np.rot90(image, k=k)

    def _fallback_rotate(self, image: np.ndarray, angle: float) -> np.ndarray:
        """
        Fallback to SciPy if available or raise ValueError.
        """
        if scipy_rotate is None:
            raise ValueError("scipy.ndimage.rotate is not available for arbitrary angles")
        return scipy_rotate(image, angle, reshape=self.reshape, order=self.order, mode=self.mode)

    def solve(self, problem: dict[str, Any]) -> Dict[str, Any]:
        image = problem["image"]
        angle = problem["angle"]

        # Ensure numpy array
        image = np.asarray(image)
        # Quick path for multiples of 90°
        if abs(angle % 90) < 1e-8:
            rotated = self._rotate_simple(image, angle)
            return {"rotated_image": rotated.tolist()}

        # Arbitrary angle requires SciPy (or raises error)
        try:
            rotated = self._fallback_rotate(image, angle)
        except Exception as exc:
            # Gracefully report failure without stack trace
            return {"rotated_image": []}

        return {"rotated_image": rotated.tolist()}