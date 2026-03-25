import numpy as np
import scipy.odr as odr


class Solver:
    def solve(self, problem, **kwargs) -> dict[str, Any]:
        """
        Fit a weighted orthogonal distance regression (ODR) model to the data.
        The model is linear: y = a * x + b.  We use scipy.odr which implements
        the weighted orthogonal sum of squares with the given standard errors.
        """
        # Convert inputs to numpy arrays
        x = np.asarray(problem["x"], dtype=float)
        y = np.asarray(problem["y"], dtype=float)
        sx = np.asarray(problem["sx"], dtype=float)
        sy = np.asarray(problem["sy"], dtype=float)

        # Create the ODR data structure and linear model
        data = odr.RealData(x, y=y, sx=sx, sy=sy)
        linear_model = odr.Model(lambda B, x: B[0] * x + B[1])

        # Initial guess for the parameters [slope, intercept]
        beta0 = [0.0, 1.0]

        # Run the ODR algorithm
        odr_result = odr.ODR(data, linear_model, beta0=beta0).run()

        # Return the fitted parameters as a list
        return {"beta": odr_result.beta.tolist()}
