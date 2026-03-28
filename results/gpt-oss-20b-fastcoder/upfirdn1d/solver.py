import numpy as np
from typing import Any

class Solver:
    def solve(self, problem: list) -> list:
        """Compute the upfirdn operation for each problem definition in the list."""
        results = []
        for h, x, up, down in problem:
            # Convert inputs to NumPy arrays (ensuring contiguous memory)
            h = np.asarray(h)
            x = np.asarray(x)

            # Up-sample `x` by inserting zeros.
            up_len = len(x) * up
            upx = np.empty(up_len, dtype=x.dtype)
            upx.fill(0)
            upx[::up] = x

            # Full convolution of upsampled `x` with filter `h`.
            conv = np.convolve(upx, h, mode="full")

            # Down-sample the convolution result.
            results.append(conv[::down])

        return results