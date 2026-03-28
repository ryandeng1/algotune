from typing import Any
import sympy


class Solver:
    def solve(self, problem: dict[str, int]) -> dict[str, int]:
        composite_val = problem["composite"]
        factors = [prime for prime, exp in sympy.factorint(composite_val).items() for _ in range(exp)]
        if len(factors) != 2:
            raise ValueError(f"Expected 2 factors, but got {len(factors)}.")
        p, q = sorted(factors)
        return {"p": p, "q": q}