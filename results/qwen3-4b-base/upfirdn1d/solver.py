import numpy as np

class Solver:
    def solve(self, problem: list) -> list:
        results = []
        for h, x, up, down in problem:
            x_up = np.zeros(len(x) * up)
            x_up[::up] = x
            y = np.convolve(x_up, h, mode='full')
            y_down = y[::down]
            results.append(y_down)
        return results
