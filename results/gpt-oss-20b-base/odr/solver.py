# solver.py
import numpy as np
import scipy.odr as odr

class Solver:
    """Optimised ODR solver using scipy.odr."""
    
    def solve(self, problem: dict[str, object]) -> dict[str, list[float]]:
        """
        Fit a weighted linear model y = b0 * x + b1 using scipy.odr,
        accepting x, y and their uncertainties sx, sy in the problem dict.
        """
        # Use numpy arrays without dtype conversion overhead
        x = np.asarray(problem['x'], dtype=float, order='C')
        y = np.asarray(problem['y'], dtype=float, order='C')
        sx = np.asarray(problem['sx'], dtype=float, order='C')
        sy = np.asarray(problem['sy'], dtype=float, order='C')

        # Prepare data and model once
        data = odr.RealData(x, y=y, sx=sx, sy=sy)
        model = odr.Model(lambda B, x: B[0] * x + B[1])

        # Run ODR with a reasonable initial guess
        output = odr.ODR(data, model, beta0=[0.0, 1.0]).run()

        return {'beta': output.beta.tolist()}