from typing import Any, Dict
import numpy as np

class Solver:
    # Use :grid, :mode etc similar to scipy's default
    mode = "nearest"
    order = 0  # nearest‑neighbour

    def _nearest_zoom(self, image: np.ndarray, factor: float) -> np.ndarray:
        """Fast integer or non‑integer zoom using nearest neighbor."""
        if factor == 1:
            return image.copy()
        # Integer scaling handled with np.kron for speed
        if factor.is_integer():
            k = int(factor)
            return np.kron(image, np.ones((k, k), dtype=image.dtype))
        # Non‑integer scaling: build coordinate grid and use nearest indexing
        src_h, src_w = image.shape[:2]
        tgt_h, tgt_w = int(round(src_h * factor)), int(round(src_w * factor))
        row_idx = np.rint(np.linspace(0, src_h - 1, tgt_h)).astype(int)
        col_idx = np.rint(np.linspace(0, src_w - 1, tgt_w)).astype(int)
        return image[row_idx[:, None], col_idx]

    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        image = problem["image"]
        zoom_factor = problem["zoom_factor"]
        try:
            zoomed_image = self._nearest_zoom(image, Zoom := float(zoom_factor))
        except Exception:
            return {"zoomed_image": np.array([])}
        return {"zoomed_image": zoomed_image}