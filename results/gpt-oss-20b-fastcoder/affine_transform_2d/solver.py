import numpy as np
from scipy.ndimage import affine_transform

class Solver:
    def __init__(self):
        self._mode = 'constant'
        self._order = 3

    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Applies a 2‑D affine transformation to the input image.

        The function assumes that ``problem`` contains two keys:
        * ``image``  – a NumPy array representing the image.
        * ``matrix`` – a 2×3 affine matrix (inverse transformation).

        Returns a dictionary containing the transformed image under
        the key ``"transformed_image"``. If the transformation can’t be
        performed, an empty list is returned instead.
        """
        image = np.asarray(problem.get('image', []))
        matrix = np.asarray(problem.get('matrix', []))

        # Fast path: if image or matrix are empty, return empty list
        if image.size == 0 or matrix.size == 0:
            return {'transformed_image': []}

        try:
            transformed = affine_transform(
                image,
                matrix,
                order=self._order,
                mode=self._mode
            )
        except Exception:
            # Any problem during the transform => return empty list
            return {'transformed_image': []}

        return {'transformed_image': transformed}