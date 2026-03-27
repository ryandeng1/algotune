import numpy as np
from typing import Any, Dict

class Solver:
    """
    A lightweight replacement for the original solver that does not depend
    on scipy.  It supports integer shifts exactly and a simple nearest‑
    neighbour interpolation for fractional shifts.
    """

    # Configurable interpolation mode: "nearest", "linear", "cubic" (fallback to nearest)
    mode: str = "nearest"

    def _shift_integer(self, image: np.ndarray, shift: tuple[int, int]) -> np.ndarray:
        """Shift the array by an integer number of pixels using np.roll."""
        return np.roll(image, shift, axis=(0, 1))

    def _shift_fractional(self, image: np.ndarray, shift: tuple[float, float]) -> np.ndarray:
        """
        Apply a small sub‑pixel shift using nearest‑neighbour interpolation.
        For a more accurate result scipy would be required.
        """
        h, w = image.shape[:2]
        # Create coordinate grid
        y, x = np.arange(h), np.arange(w)
        X, Y = np.meshgrid(x, y, indexing="xy")
        # Shift coordinates
        Xs = X - shift[1]
        Ys = Y - shift[0]
        # Nearest neighbour
        Xn = np.clip(np.round(Xs).astype(int), 0, w - 1)
        Yn = np.clip(np.round(Ys).astype(int), 0, h - 1)
        if image.ndim == 3:
            return image[Yn, Xn]
        else:
            return image[Yn, Xn]

    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Shift the input image by the given shift vector.

        The shift vector may contain fractional components.  Integer components
        are handled by a fast np.roll, while fractional components are
        approximated with nearest‑neighbour interpolation.

        :param problem: Dictionary with keys:
            - "image": 2D or 3D numpy array to shift.
            - "shift": Tuple of (row_shift, col_shift), floats allowed.
        :return: Dictionary with key "shifted_image" containing the shifted array
                 or an empty list if the operation fails.
        """
        try:
            image = problem["image"]
            shift = problem["shift"]

            # Ensure image is a NumPy array
            if not isinstance(image, np.ndarray):
                image = np.asarray(image)

            # Validate shift vector
            if not (isinstance(shift, (tuple, list)) and len(shift) == 2):
                raise ValueError("Shift must be a two‑element tuple or list.")

            # Separate integer and fractional parts
            int_shift = tuple(int(np.floor(s)) for s in shift)
            frac_shift = tuple(s - np.floor(s) for s in shift)

            # Apply integer shift
            shifted = self._shift_integer(image, int_shift)

            # Apply fractional shift if needed
            if any(abs(f) > 1e-12 for f in frac_shift):
                shifted = self._shift_fractional(shifted, frac_shift)

            return {"shifted_image": shifted}

        except Exception:
            # Return an empty list to indicate failure, matching the original API
            return {"shifted_image": []}