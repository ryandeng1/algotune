import numpy as np
from typing import List, Tuple, Any

class Solver:
    def solve(self, problem: List[Tuple[np.ndarray, np.ndarray, int, int]]) -> List[np.ndarray]:
        """
        Compute the upfirdn operation for each problem definition in the list
        without using SciPy.

        Parameters
        ----------
        problem : List[Tuple[h, x, up, down]]
            * h   : FIR filter coefficients (1‑D ndarray)
            * x   : input signal (1‑D ndarray)
            * up  : upsample factor (int)
            * down: downsample factor (int)

        Returns
        -------
        List[np.ndarray]
            Results of the upfirdn operation for each tuple.
        """
        results = []

        for h, x, up, down in problem:
            # 1. Upsample x by inserting (up-1) zeros between samples
            if up != 1:
                up_len = len(x) * up - (up - 1)
                upx = np.empty(up_len, dtype=x.dtype)
                upx[::up] = x
                upx[1::up] = 0
            else:
                upx = x

            # 2. Convolve with FIR filter h (full mode)
            #    Using np.convolve is efficient for 1‑D signals
            conv = np.convolve(upx, h, mode='full')

            # 3. Downsample by taking every `down`-th sample
            if down != 1:
                res = conv[::down]
            else:
                res = conv

            results.append(res)

        return results