from typing import Any, Dict, List
import numpy as np

class Solver:
    """
    Optimised 2D image zoomer.
    Supports:
      * nearest‑neighbour (`order=0`) – fastest, implemented with numpy repeat
      * bilinear (`order=1`) – efficient implementation based on
        linear interpolation
    """
    order: int = 1                           # default to bilinear
    mode: str = "nearest"                    # only used for borders in bilinear

    def _nearest_neighbour(self, img: np.ndarray, zoom: float) -> np.ndarray:
        """Nearest‑neighbour zoom using numpy repeat."""
        kr, kc = int(round(zoom)), int(round(zoom))
        return img.repeat(kr, axis=0).repeat(kc, axis=1)

    def _bilinear(self, img: np.ndarray, zoom: float) -> np.ndarray:
        """Bilinear zoom using efficient padding + vectorised interpolation."""
        if zoom == 1.0:
            return img.copy()

        # calculate output shape
        h, w = img.shape[:2]
        h_out, w_out = int(round(h * zoom)), int(round(w * zoom))

        # generate output grid coordinates
        row_src = np.linspace(0, h - 1, h_out)
        col_src = np.linspace(0, w - 1, w_out)
        row, col = np.meshgrid(row_src, col_src, indexing="ij")

        # floor and ceil indices
        r0 = np.floor(row).astype(int)
        r1 = np.clip(r0 + 1, 0, h - 1)
        c0 = np.floor(col).astype(int)
        c1 = np.clip(c0 + 1, 0, w - 1)

        # fractional parts
        dr = row - r0
        dc = col - c0

        # gather pixels
        if img.ndim == 2:  # grayscale
            img = img[..., None]
        top_left = img[r0, c0]
        top_right = img[r0, c1]
        bottom_left = img[r1, c0]
        bottom_right = img[r1, c1]

        # bilinear interpolation
        top = top_left * (1 - dc)[..., None] + top_right * dc[..., None]
        bottom = bottom_left * (1 - dc)[..., None] + bottom_right * dc[..., None]
        interpolated = top * (1 - dr)[..., None] + bottom * dr[..., None]

        return interpolated.squeeze()

    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        img = problem["image"]
        zoom_factor = float(problem["zoom_factor"])

        try:
            if self.order == 0:
                zoomed = self._nearest_neighbour(img, zoom_factor)
            else:
                zoomed = self._bilinear(img, zoom_factor)
        except Exception:
            return {"zoomed_image": []}

        return {"zoomed_image": zoomed}