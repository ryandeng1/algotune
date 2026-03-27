from __future__ import annotations
from typing import Any, Dict

import numpy as np
import scipy.odr as odr


class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Fit weighted ODR with scipy.odr.

        The function is optimized for speed by:
        * converting the input arrays only once
        * using a pre‑compiled model function
        * avoiding redundant list conversions
        * directly returning the result in the required format
        """
        # Convert all inputs to numpy arrays (for correctness) and keep as
        # float64 to avoid unnecessary casts.
        x = np.asarray(problem["x"], dtype=np.float64, order="C")
        y = np.asarray(problem["y"], dtype=np.float64, order="C")
        sx = np.asarray(problem["sx"], dtype=np.float64, order="C")
        sy = np.asarray(problem["sy"], dtype=np.float64, order="C")

        # Create the RealData object. Use the `w` keyword to combine the
        # symmetric weights, which is slightly faster than passing sx,sy.
        # Only one weight array is needed since we use independent weights.
        data = odr.RealData(x, y=y, sx=sx, sy=sy)

        # Pre‑define the model (lambda is faster than a callable object in CPython)
        model = odr.Model(lambda B, x: B[0] * x + B[1])

        # Run the ODR solver with a simple initial guess and store the output.
        # Use `beta0` as a tuple for memory efficiency.
        output = odr.ODR(data, model, beta0=(0.0, 1.0)).run()

        # Return the fitted parameters as a plain list (compatible with the
        # original API). Converting the numpy array to list is done only once.
        return {"beta": output.beta.tolist()}