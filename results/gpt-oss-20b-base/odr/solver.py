import numpy as np
import scipy.odr as odr

class Solver:
    def solve(self, problem):
        # directly use numpy arrays without explicit copying
        x, y, sx, sy = (
            np.asarray(problem['x']),
            np.asarray(problem['y']),
            np.asarray(problem['sx']),
            np.asarray(problem['sy']),
        )
        data = odr.RealData(x, y=y, sx=sx, sy=sy)
        model = odr.Model(lambda beta, x: beta[0] * x + beta[1])
        output = odr.ODR(data, model, beta0=[0.0, 1.0]).run()
        return {'beta': list(output.beta)}