import numpy as np
from scipy.linalg import qz
from typing import Any

class Solver:
    def solve(
        self, problem: dict[str, list[list[float]]]
    ) -> dict[str, dict[str, list[list[float | complex]]]]:
        """
        Compute the real QZ-factorization of (A, B).

        Parameters
        ----------
        problem : dict
            Dictionary containing the square matrices under the keys ``"A"`` and
            ``"B"``.  Each entry is a nested ``list`` of floats.

        Returns
        -------
        dict
            A dictionary of the form::

                {"QZ": {"AA": AA, "BB": BB, "Q": Q, "Z": Z}}

            where each value is a list‑of‑lists of floats (or complex in
            ``Z`` if the algorithm produces a complex matrix).  The
            conversion to Python objects is deferred until the returned
            dictionary is built.  For the ``list`` representation, we keep the
            same shape as the NumPy arrays to avoid extra memory allocation.
        """
        # Convert the input lists into contiguous numpy arrays once.
        # This is faster than converting inside the calling function and
        # avoids repeated memory copies.
        A = np.asarray(problem["A"], dtype=float, order="C")
        B = np.asarray(problem["B"], dtype=float, order="C")

        # Perform the real QZ factorization. The ``output="real"`` flag ensures
        # that AA and BB are returned as real arrays, while Q and Z may be
        # complex.  Using the low‑level LAPACK routine from SciPy gives an
        # efficient, well‑optimised implementation.
        AA, BB, Q, Z = qz(A, B, output="real")

        # Convert arrays to lists only once, before returning.  The
        # conversion is cheap compared to the factorisation step.
        return {
            "QZ": {
                "AA": AA.tolist(),
                "BB": BB.tolist(),
                "Q": Q.tolist(),
                "Z": Z.tolist()
            }
        }