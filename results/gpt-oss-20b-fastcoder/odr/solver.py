import numpy as np
from scipy.odr import ODR, RealData, Model

def solve(problem: dict) -> dict:
    """Fit weighted ODR with scipy.odr."""
    x = np.asarray(problem["x"])
    y = np.asarray(problem["y"])
    sx = np.asarray(problem["sx"])
    sy = np.asarray(problem["sy"])
    data = RealData(x, y=y, sx=sx, sy=sy)
    model = Model(lambda B, x: B[0] * x + B[1])
    output = ODR(data, model, beta0=[0.0, 1.0]).run()
    return {"beta": output.beta.tolist()}