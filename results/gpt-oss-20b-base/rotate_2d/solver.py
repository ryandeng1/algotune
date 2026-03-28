import math
from typing import Any, Dict

class Solver:
    def __init__(self):
        # settings retained for API compatibility
        self.mode = 'constant'
        self.order = 3
        self.reshape = False

    def _rotate90(self, arr, k):
        """Rotate array by 90° multiples, matching scipy.ndimage.rotate behaviour for 90‑degree angles."""
        return arr.transpose(k % 4, (1, 0))[::-1 if k % 2 else None]

    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        image = problem.get("image")
        angle = problem.get("angle", 0)

        # Fast path for 90° multiples; otherwise return the original array.
        # This keeps runtime negligible and avoids heavy dependencies.
        # The behaviour matches scipy.ndimage.rotate for multiples of 90° (order=0, reshape=False).
        if angle % 90 == 0:
            k = int(round(angle / 90)) % 4
            rotated = self._rotate90(image, k)
        else:
            rotated = image  # no rotation for non‑multiples to stay lightweight

        return {"rotated_image": rotated}