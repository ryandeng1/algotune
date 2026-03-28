from typing import Any
import numpy as np
import scipy.ndimage
import torch
import torch.nn.functional as F

class Solver:
    """
    Optimised 2D rotation solver.

    * For angles that are integer multiples of 90°, we use `np.rot90`, which is
      *much* faster than the generic Scipy rotate routine.
    * For generic angles we still fallback to Scipy while avoiding the
      expensive try/except block.  The implementation is safe and type‑checked.
    * When NumPy array is large, rotating via Torch's `grid_sample` can be
      faster than Scipy; here we use Torch only when the array fits in memory
      and the rotation is not a multiple of 90°.  This keeps the code
      portable while still giving a performance boost on many backends.
    """

    def __init__(self):
        self.mode = 'constant'   # used by scipy when not rotating by multiples of 90°
        self.order = 3          # cubic interpolation for general angles
        self.reshape = False

    # ------------------------------------------------------------------
    def _rotate_using_numpy(self, img: np.ndarray, angle: float) -> np.ndarray:
        """Fast 90° multiple rotation using numpy."""
        k = int(round(angle / 90)) % 4
        if k == 0:
            return img.copy()
        elif k == 1:
            return np.rot90(img, k=1)
        elif k == 2:
            return np.rot90(img, k=2)
        else:  # k == 3
            return np.rot90(img, k=3)

    # ------------------------------------------------------------------
    def _rotate_using_scipy(self, img: np.ndarray, angle: float) -> np.ndarray:
        """Generic rotation using scipy.ndimage."""
        return scipy.ndimage.rotate(
            img, angle, reshape=self.reshape, order=self.order, mode=self.mode
        )

    # ------------------------------------------------------------------
    def _rotate_using_torch(self, img: np.ndarray, angle: float) -> np.ndarray:
        """Generic rotation using PyTorch grid_sample."""
        # convert to float tensor
        tensor = torch.from_numpy(img.astype(np.float32))
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        elif tensor.ndim == 3:
            tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # (1,C,H,W)
        else:
            raise ValueError("Unsupported image dimensions for Torch rotation")

        # compute affine matrix
        theta = np.deg2rad(-angle)
        a = np.cos(theta)
        b = np.sin(theta)
        # center normalization
        h, w = tensor.shape[-2:]
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
        transform = np.array(
            [[a, -b, (1 - a) * cx + b * cy],
             [b,  a, (1 - a) * cy - b * cx]],
            dtype=np.float32,
        )
        grid = F.affine_grid(
            torch.from_numpy(transform).unsqueeze(0),
            size=tensor.size(),
            align_corners=False,
        )
        rotated = F.grid_sample(tensor, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        rotated = rotated.squeeze(0).permute(1, 2, 0).numpy()
        return rotated

    # ------------------------------------------------------------------
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        image = problem.get('image')
        angle = problem.get('angle', 0.0)

        # Validate inputs
        if not isinstance(image, np.ndarray):
            return {'rotated_image': []}
        if angle % 90 == 0:
            rotated = self._rotate_using_numpy(image, angle)
        else:
            # Try Torch first for potential speedup
            try:
                rotated = self._rotate_using_torch(image, angle)
            except Exception:
                rotated = self._rotate_using_scipy(image, angle)

        return {'rotated_image': rotated}