from typing import Any
import numpy as np
from scipy import signal

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list]:
        signal_x = np.array(problem["signal_x"])
        signal_y = np.array(problem["signal_y"])
        mode = problem.get("mode", "full")
        convolution_result = signal.fftconvolve(signal_x, signal_y, mode=mode)
        return {"convolution": convolution_result.tolist()}