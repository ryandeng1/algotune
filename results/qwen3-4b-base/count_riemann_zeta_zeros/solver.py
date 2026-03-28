from typing import Any
from mpmath import mp

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        t = problem["t"]
        result = mp.nzeros(mp.zeta, 0.5 - 1j*t, 0.5 + 1j*t)
        return {"result": result}