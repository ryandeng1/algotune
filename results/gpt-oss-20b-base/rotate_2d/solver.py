import numpy as np
try:
    import scipy.ndimage
except ImportError:
    scipy = None

def solve(problem: dict) -> dict:
    """
    Rotate an image without the expense of exception handling.
    It uses scipy.ndimage.rotate if available; otherwise falls back
    to a fast numpy-based rotation for 90-deg multiples.

    Parameters
    ----------
    problem : dict
        Must contain keys:
            "image" : 2-D :class:`numpy.ndarray`
            "angle" : float

    Returns
    -------
    dict
        {"rotated_image": rotated array}
    """
    image: np.ndarray = problem["image"]
    angle: float = float(problem["angle"])
    # Normalize angle to [-180, 180] for consistency
    angle = (angle + 180) % 360 - 180

    # Fast path for multiples of 90°
    if abs(angle % 90) < 1e-9:
        k = int(round(angle / 90)) % 4
        rotated = np.rot90(image, k, axes=(0, 1))
    elif scipy is not None:
        rotated = scipy.ndimage.rotate(
            image, angle, reshape=False, order=1, mode='nearest'
        )
    else:
        # Generic rotation via interpolation grid – simple but slower
        h, w = image.shape
        y, x = np.indices((h, w))
        cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
        ry = np.cos(np.deg2rad(angle))
        sx = -np.sin(np.deg2rad(angle))
        cx_ = np.sin(np.deg2rad(angle))
        sy = np.cos(np.deg2rad(angle))

        # Translate to centre, rotate, translate back
        x0 = (x - cx) * ry + (y - cy) * cx_ + cx
        y0 = -(x - cx) * sx + (y - cy) * sy + cy

        # Bilinear interpolation
        x0_floor = np.floor(x0).astype(int)
        y0_floor = np.floor(y0).astype(int)
        xp = np.clip(x0_floor + 1, 0, w - 1)
        yp = np.clip(y0_floor + 1, 0, h - 1)

        x0_floor = np.clip(x0_floor, 0, w - 1)
        y0_floor = np.clip(y0_floor, 0, h - 1)

        wx = x0 - x0_floor
        wy = y0 - y0_floor

        rotated = (
            (1 - wx) * (1 - wy) * image[y0_floor, x0_floor]
            + wx * (1 - wy) * image[y0_floor, xp]
            + (1 - wx) * wy * image[yp, x0_floor]
            + wx * wy * image[yp, xp]
        )

    return {"rotated_image": rotated}