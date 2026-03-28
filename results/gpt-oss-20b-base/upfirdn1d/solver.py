import numpy as np
from typing import Any, List, Tuple

class Solver:
    def solve(self, problem: List[Tuple[Any, Any, int, int]]) -> List[np.ndarray]:
        """
        Compute the upfirdn operation for each problem definition in the list.
        Parameters
        ----------
        problem : list of tuples (h, x, up, down)
            - h : 1‑D array-like filter coefficients
            - x : 1‑D array-like input signal
            - up : integer up‑sampling factor (>=1)
            - down : integer down‑sampling factor (>=1)
        Returns
        -------
        list of np.ndarray
            The upfirdn results for each tuple
        """
        results = []

        for h, x, up, down in problem:
            # Convert to numpy arrays
            h = np.asarray(h, dtype=np.float64)
            x = np.asarray(x, dtype=np.float64)

            # Upsample by inserting zeros
            if up > 1:
                up_x_len = len(x) * up
                up_x = np.empty(up_x_len, dtype=x.dtype)
                up_x[::up] = x
            else:
                up_x = x

            # Linear convolution (full)
            res = np.convolve(up_x, h, mode='full')

            # Downsample
            if down > 1:
                res = res[::down]

            results.append(res)

        return results