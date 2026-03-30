from typing import Any
import numpy as np
import scipy.odr as odr


class Solver:
    """
    Fast ODR solver.

    The implementation avoids intermediate Python objects and uses Numpy
    views to keep the amount of data copied to a minimum.
    """

    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        # Convert everything to contiguous float64 arrays once
        x = np.asarray(problem["x"], dtype=np.float64, order="C")
        y = np.asarray(problem["y"], dtype=np.float64, order="C")
        sx = np.asarray(problem["sx"], dtype=np.float64, order="C")
        sy = np.asarray(problem["sy"], dtype=np.float64, order="C")

        # Build the data and model objects used by scipy.odr
        data = odr.RealData(x, y=y, sx=sx, sy=sy)
        # Linear model B[0]*x + B[1]
        model = odr.Model(lambda B, x: B[0] * x + B[1])

        # Perform the fit; beta0 is a good initial guess
        odr_obj = odr.ODR(data, model, beta0=[0.0, 1.0])
        output = odr_obj.run()

        # Return result in the expected format
        return {"beta": output.beta.tolist()}