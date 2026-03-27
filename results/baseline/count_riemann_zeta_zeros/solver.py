from typing import Any
from mpmath import mp


class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """Count zeta zeros along critical strip with imaginary part <= t.

        Using `mpmath.mp.nzeros`.
        """
        result = mp.nzeros(problem["t"])
        return {"result": result}
