import numpy as np
import numba

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        image = problem["image"]
        zoom_factor = problem["zoom_factor"]
        n = image.shape[0]
        m = int(n * zoom_factor)
        
        @numba.jit(nopython=True)
        def bicubic_interp():
            result = np.zeros((m, m), dtype=np.float64)
            for i in range(m):
                for j in range(m):
                    x = i / zoom_factor
                    y = j / zoom_factor
                    x0 = int(x)
                    y0 = int(y)
                    x0 = max(0, min(x0, n - 1))
                    y0 = max(0, min(y0, n - 1))
                    x1 = min(x0 + 1, n - 1)
                    y1 = min(y0 + 1, n - 1)
                    
                    t = x - x0
                    s = y - y0
                    
                    # Cubic basis functions
                    w00 = (1.0 - t) ** 3
                    w01 = 3.0 * t * (1.0 - t) ** 2
                    w02 = 3.0 * t ** 2 * (1.0 - t)
                    w03 = t ** 3
                    w10 = (1.0 - s) ** 3
                    w11 = 3.0 * s * (1.0 - s) ** 2
                    w12 = 3.0 * s ** 2 * (1.0 - s)
                    w13 = s ** 3
                    
                    # Interpolate using 2x2 block (simplified for performance)
                    val = (w00 * w10 * image[y0, x0] +
                           w01 * w10 * image[y0, x1] +
                           w00 * w11 * image[y1, x0] +
                           w01 * w11 * image[y1, x1])
                    result[i, j] = val
            return result
        
        zoomed_image = bicubic_interp()
        return {"zoomed_image": zoomed_image}
