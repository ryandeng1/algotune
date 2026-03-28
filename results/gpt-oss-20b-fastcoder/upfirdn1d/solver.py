import numpy as np

class Solver:
    def solve(self, problem: list) -> list:
        """
        Compute the upfirdn operation for each problem definition in the list
        without relying on scipy, using efficient NumPy operations.
        """
        results = []

        for h, x, up, down in problem:
            # ensure arrays
            h = np.asarray(h, dtype=float)
            x = np.asarray(x, dtype=float)

            # --- upsample ---
            if up == 1:
                x_up = x
            else:
                # insert (up-1) zeros between samples
                n = x.size
                n_up = n * up
                x_up = np.empty(n_up, dtype=x.dtype)
                x_up[::up] = x

            # --- convolution ---
            y = np.convolve(h, x_up)

            # --- downsample ---
            if down == 1:
                y_down = y
            else:
                y_down = y[::down]

            results.append(y_down)

        return results