from typing import Any, Dict
import numpy as np
import scipy.ndimage

# NOTE: This implementation is deliberately simple yet fast.  The surrounding
# infrastructure guarantees that all inputs are valid NumPy arrays with a
# numeric dtype and that `zoom_factor` is a positive scalar or a 2‑tuple.
# Because those pre‑conditions are known, we can drop the generic exception
# handling that used to surround the call to `scipy.ndimage.zoom`.  This
# removes an unnecessary dynamic dispatch path and eliminates the jitter of
# throwing/handling an exception for the common (successful) case.

class Solver:

    # Using a class attribute keeps the configuration immutable and
    # thread safe.  It also reduces instance allocation time.
    mode = "constant"
    order = 3

    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform a 2‑D zoom operation on the supplied image.

        Parameters
        ----------
        problem:
            Dictionary containing keys:
            - 'image': 2‑D or 3‑D NumPy array (image data)
            - 'zoom_factor': float or sequence of two floats

        Returns
        -------
        dict
            {'zoomed_image': <zoomed array>}
        """
        # Retrieve the inputs once.
        img = problem["image"]
        zf = problem["zoom_factor"]

        # Fast path: use `scipy.ndimage.zoom` directly.  The function
        # returns a new array, so we avoid any extra copying or
        # intermediate allocations.
        zoomed = scipy.ndimage.zoom(img, zf, order=self.order, mode=self.mode)

        # Wrap the result in the expected dictionary.
        return {"zoomed_image": zoomed}