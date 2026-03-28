from typing import Any
import sympy

class Solver:
    def solve(self, problem: dict[str, int]) -> dict[str, int]:
        composite_val = problem["composite"]
        factor_dict = sympy.factorint(composite_val)
        total_exponents = sum(factor_dict.values())
        if total_exponents != 2:
            raise ValueError(f"Expected 2 factors, but got {total_exponents}.")
        primes = sorted(factor_dict.keys())
        return {"p": primes[0], "q": primes[0] if len(primes) == 1 else primes[1]}