from typing import Any
from sympy.ntheory.residue_ntheory import discrete_log


class Solver:
    def solve(self, problem: dict[str, int]) -> dict[str, int]:
        p = problem["p"]
        g = problem["g"]
        h = problem["h"]
        
        # Handle trivial cases for performance
        if h == 1:
            return {"x": 0}
        if g == 1:
            return {"x": 0}
        
        return {"x": discrete_log(p, h, g)}