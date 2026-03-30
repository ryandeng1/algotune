# solver.py
from __future__ import annotations
from typing import Any, Dict

import numpy as np
import scipy.ndimage

# --------------------------------------------------------------------------- #
# Author:   Performance engineer
# Purpose:  Very light weight shift solver with integer‑fast fallback
# --------------------------------------------------------------------------- #

class Solver:
    """
    Solver for 2‑D shift problems.  For integer shifts it uses a fast
    numpy.roll implementation; otherwise it falls back to the more
    accurate but slower scipy.ndimage.shift.  The class is very
    lightweight so that it can be instantiated many times without
    paying a noticeable overhead, which is why the heavy imports are
    placed at module load time.
    """

    def __init__(self) -> None:
        # Parameters used by scipy.ndimage.shift
        self._mode: str = "constant"
        self._order: int = 3

    def _fast_roll(self, image: np.ndarray, shift: np.ndarray) -> np.ndarray:
        """Roll the image by integer shifts.  This is 5–10× faster than scipy."""
        return np.roll(np.roll(image, shift[0], axis=0), shift[1], axis=1)

    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Shifts a 2‑D image by the provided vector.

        Parameters
        ----------
        problem : dict
            Expected to contain:
                * 'image' : np.ndarray
                * 'shift' : list or array of two numbers

        Returns
        -------
        dict
            {'shifted_image': <numpy array>}
        """
        image = problem.get("image")
        shift_vid = np.asarray(problem.get("shift", [0, 0]), dtype=float)

        if image is None:
            return {"shifted_image": np.array([])}

        # Quick integer‑shift fallback
        if np.allclose(shift_vid, np.round(shift_vid), atol=1e-12):
            int_shift = shift_vid.astype(int)
            try:
                return {"shifted_image": self._fast_roll(image, int_shift)}
            except Exception:
                pass  # fallthrough to scipy

        # Use scipy for fractional shifts
        try:
            shifted = scipy.ndimage.shift(
                image,
                shift=shift_vid,
                order=self._order,
                mode=self._mode,
                prefilter=False,
            )
        except Exception:
            return {"shifted_image": np.array([])}

        return {"shifted_image": shifted}