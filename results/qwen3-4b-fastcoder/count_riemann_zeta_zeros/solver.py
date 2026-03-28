from typing import Any
from mpmath import mp

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        t = problem["t"]
        count = 0
        n = 0
        while True:
            try:
                zero = mp.zeta_zeros(n + 1)
                if zero.imag <= t:
                    count += 1
                else:
                    break
            except:
                break
            n += 1
        return {"result": count}