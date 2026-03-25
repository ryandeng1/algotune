import math
import numba
from numba import njit

class Solver:
    @njit
    def _baby_step_giant_step(self, p, g, h):
        n = p - 1
        m = int(math.isqrt(n)) + 1
        
        baby_steps = {}
        current = 1
        for j in range(m):
            baby_steps[current] = j
            current = (current * g) % p
        
        g_m = pow(g, m, p)
        g_inv_m = pow(g_m, p - 2, p)
        
        current = h
        for i in range(m + 1):
            if current in baby_steps:
                j = baby_steps[current]
                return i * m + j
            current = (current * g_inv_m) % p
        return None

    def solve(self, problem: dict[str, int]) -> dict[str, int]:
        p = problem["p"]
        g = problem["g"]
        h = problem["h"]
        x = self._baby_step_giant_step(p, g, h)
        return {"x": x}
