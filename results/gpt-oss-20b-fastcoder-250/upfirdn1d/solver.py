import numpy as np
import math
from typing import Any, List, Tuple

class Solver:
    def solve(self, problem: List[Tuple[np.ndarray, np.ndarray, int, int]]) -> List[np.ndarray]:
        """
        Efficiently perform upfirdn (upsampling → filtering → downsampling)
        for each instance in `problem`. Each element of `problem` is a tuple
        (h, x, up, down).

        Parameters
        ----------
        problem : list of tuples
            Each tuple contains a 1‑D filter `h`, a 1‑D input signal `x`,
            an upsampling factor `up` (int > 0), and a downsampling factor `down`
            (int > 0).

        Returns
        -------
        list of np.ndarray
            The resulting 1‑D arrays after upfirdn for each problem instance.
            The result list has the same length and order as `problem`.
        """
        results: List[np.ndarray] = []

        for h, x, up, down in problem:
            # Convert to 1‑D float arrays
            h = np.asarray(h, dtype=np.float64).ravel()
            x = np.asarray(x, dtype=np.float64).ravel()

            # Avoid trivial cases
            if x.size == 0 or h.size == 0:
                results.append(np.array([], dtype=np.float64))
                continue

            if up == 1 and down == 1:
                # No resampling, just convolution
                results.append(np.convolve(x, h, mode='full'))
                continue

            # Upsample: create array of zeros with inserted samples
            up_len = x.size * up
            y = np.empty(up_len, dtype=np.float64)
            if up == 1:
                y = x.copy()
            else:
                y.fill(0.0)
                y[::up] = x

            # Convolve with the filter
            conv = np.convolve(y, h, mode='full')

            # Downsample by taking every `down`‑th sample
            out = conv[::down]

            results.append(out)

        return results
