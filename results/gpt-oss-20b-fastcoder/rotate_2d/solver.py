from typing import Any
import scipy.ndimage


class Solver:
    """
    Fast implementation of a 2‑D rotation using scipy.ndimage.rotate.
    """

    __slots__ = ("mode", "order", "reshape")

    def __init__(self) -> None:
        # Caching the common keyword arguments avoids re‑creating the strings
        # on every call, which makes the function noticeably faster for many
        # successive invocations.
        self.mode = "constant"
        self.order = 3
        self.reshape = False

    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Rotate the supplied image by the given angle.

        Parameters
        ----------
        problem :
            Dictionary containing the keys ``image`` (NumPy array) and
            ``angle`` (float or int).

        Returns
        -------
        dict
            Dictionary with a single key ``rotated_image`` whose value is
            the rotated NumPy array.
        """
        # Unpack the problem dict once and avoid attribute lookups.
        image = problem["image"]
        angle = problem["angle"]

        # The rotate function is highly optimized in C, so we just forward the
        # call directly. Any exception raised here will propagate to the caller,
        # keeping the overhead minimal.
        rotated_image = scipy.ndimage.rotate(
            image, angle, reshape=self.reshape, order=self.order, mode=self.mode
        )

        return {"rotated_image": rotated_image}