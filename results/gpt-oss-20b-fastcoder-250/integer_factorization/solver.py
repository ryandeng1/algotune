import math
from typing import Any, Dict


class Solver:
    """
    Optimised factoriser.  Uses a deterministic trial‑division up to
    sqrt(n) with a step of 6 (i.e. tests 2,3 and then 6k±1).
    Works well for integers up to ~10^12 or larger if they are
    products of two primes.
    """

    @staticmethod
    def _prime_factors(n: int) -> list[int]:
        """Return the list of prime factors of n (with multiplicity)."""
        factors: list[int] = []

        # Handle small primes 2 and 3 explicitly
        while n % 2 == 0:
            factors.append(2)
            n //= 2
        while n % 3 == 0:
            factors.append(3)
            n //= 3

        # Check 6k ± 1
        f = 5
        limit = int(math.isqrt(n)) + 1
        while f <= limit:
            while n % f == 0:
                factors.append(f)
                n //= f
                limit = int(math.isqrt(n)) + 1
            while n % (f + 2) == 0:
                factors.append(f + 2)
                n //= f + 2
                limit = int(math.isqrt(n)) + 1
            f += 6

        # If remaining n is >1 it's a prime
        if n > 1:
            factors.append(n)

        return factors

    def solve(self, problem: Dict[str, int]) -> Dict[str, int]:
        """
        Robust factorise the provided composite number and return the two prime factors
        in ascending order.

        Raises ValueError if conversion fails or if the factorisation does not
        yield exactly two primes.
        """
        composite_val = problem["composite"]

        if not isinstance(composite_val, int):
            raise ValueError(f"Composite value must be an integer, got {type(composite_val).__name__}")

        if composite_val <= 1:
            raise ValueError(f"Composite must be > 1, got {composite_val}")

        factors = self._prime_factors(composite_val)

        if len(factors) != 2:
            raise ValueError(f"Expected 2 factors, but got {len(factors)}. Factors: {factors}")

        p, q = sorted(map(int, factors))
        return {"p": p, "q": q}