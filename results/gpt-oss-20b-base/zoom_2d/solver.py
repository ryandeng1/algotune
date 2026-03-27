import numpy as np
from typing import Any, Dict


class Solver:
    """Fast image zoomer – falls back to a simple repeat‑based upsample for integer factors.  
    Uses `numpy` only and a small amount of `scipy` if available."""

    # Parameters that mimic the original API – can be overridden by the caller.
    order: int = 1
    mode: str = "nearest"

    def __init__(self) -> None:
        # Import lazily so a missing SciPy installation doesn't crash the import.
        try:  # pragma: no cover - guard is trivial
            from scipy import ndimage  # type: ignore
            self._ndimage_zoom = ndimage.zoom
        except Exception:  # pragma: no cover
            self._ndimage_zoom = None

    def _int_zoom(self, image: np.ndarray, factor: float) -> np.ndarray:
        """Repeatedly expand or shrink the array for integer zoom factors."""
        if factor == 1.0:
            return image
        if factor > 1.0:
            s = int(round(factor))
            return np.repeat(np.repeat(image, s, axis=0), s, axis=1)
        # `factor` < 1.0 – use maximum integer pool below
        s = int(round(1.0 / factor))
        # Down‑sample by selecting an evenly spaced subset of rows/cols
        rows = np.linspace(0, image.shape[0] - 1, image.shape[0] // s, dtype=int)
        cols = np.linspace(0, image.shape[1] - 1, image.shape[1] // s, dtype=int)
        return image[rows[:, None], cols]

    def _scipy_zoom(
        self, image: np.ndarray, factor: float
    ) -> np.ndarray | None:
        """Attempt to use scipy.ndimage.zoom; return None if it fails."""
        if self._ndimage_zoom is None:
            return None
        try:
            return self._ndimage_zoom(
                image,
                zoom=factor,
                order=self.order,
                mode=self.mode,
            )
        except Exception:  # pragma: no cover
            return None

    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        image = problem["image"]
        zoom_factor = problem["zoom_factor"]

        # Prefer SciPy if available; fall back to integer repeat logic.
        zoomed_image = self._scipy_zoom(image, float(zoom_factor))
        if zoomed_image is None:
            zoomed_image = self._int_zoom(np.asarray(image), float(zoom_factor))

        # Ensure a numpy array is returned
        zoomed_image = np.asarray(zoomed_image)
        return {"zoomed_image": zoomed_image}