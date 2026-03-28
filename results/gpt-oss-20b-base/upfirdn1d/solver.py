from typing import Any, List, Tuple
import numpy as np
from numba import njit

@njit
def _upfirdn_numba(h, x, up, down):
    # Upsample x by inserting zeros between samples
    up_len = len(x) * up
    upsampled = np.empty(up_len, dtype=x.dtype)
    upsampled[:] = 0
    upsampled[::up] = x

    # Linear convolution
    conv_len = len(upsampled) + len(h) - 1
    conv = np.empty(conv_len, dtype=upsampled.dtype)
    for i in range(conv_len):
        s = 0
        j_min = 0 if i < len(h)-1 else i - (len(h)-1)
        j_max = min(i, len(upsampled)-1)
        for j in range(j_min, j_max+1):
            s += upsampled[j] * h[i-j]
        conv[i] = s

    # Downsample
    down_len = conv_len // down
    if conv_len % down != 0:
        down_len += 1
    res = np.empty(down_len, dtype=conv.dtype)
    for i in range(down_len):
        idx = i * down
        res[i] = conv[idx] if idx < conv_len else 0
    return res

class Solver:
    def solve(self, problem: List[Tuple[Any, Any, int, int]]) -> List[np.ndarray]:
        results = []
        for h, x, up, down in problem:
            h_np = np.asarray(h, dtype=np.float64)
            x_np = np.asarray(x, dtype=np.float64)
            res = _upfirdn_numba(h_np, x_np, up, down)
            results.append(res)
        return results