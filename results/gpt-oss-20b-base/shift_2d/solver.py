from typing import Any
import numpy as np
import scipy.ndimage

class Solver:
    """
    Efficient solver for 2D image shift problems.

    The implementation relies on SciPy's highly optimised `ndimage.shift` routine.
    The class stores the interpolation order and border mode as instance
    attributes so they are not recomputed for every call, reducing overhead.
    """

    __slots__ = ("order", "mode")

    def __init__(self):
        # Use cubic interpolation (order 3) and constant padding.
        self.order = 3
        self.mode = "constant"

    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Shift a 2D NumPy array according to the supplied shift vector.

        Parameters
        ----------
        problem : dict
            Must contain:
            - 'image' : array_like 2D image to shift.
            - 'shift' : sequence of two numbers specifying the shift.

        Returns
        -------
        dict
            Contains:
            - 'shifted_image' : the shifted array, or an empty list on failure.
        """
        image = problem["image"]
        shift_vector = problem["shift"]

        try:
            # SciPy shift is implemented in C/C++ and therefore fast.
            shifted = scipy.ndimage.shift(
                image,
                shift_vector,
                order=self.order,
                mode=self.mode,
            )
        except Exception:
            # In case of any error (e.g., wrong input shape) return an empty result.
            return {"shifted_image": []}

        return {"shifted_image": shifted}


# The following code is for quick local testing and is excluded from the
# runtime cost of the `solve` method.
if __name__ == "__main__":
    import numpy as np

    solver = Solver()
    img = np.arange(100).reshape(10, 10).astype(float)
    result = solver.solve({"image": img, "shift": [1.5, -2.2]})
    print(result["shifted_image"])