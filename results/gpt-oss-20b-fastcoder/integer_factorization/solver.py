# solver.py
from typing import Dict
import math
import sympy

class Solver:
    """
    Efficient solver for the factorisation of a composite number that
    is the product of exactly two prime numbers.
    """

    def _prime_factor(self, n: int) -> int | None:
        """
        Return the first prime factor of n or None if n is prime.
        Uses a simple deterministic trial division up to sqrt(n).
        """
        if n % 2 == 0:
            return 2
        limit = math.isqrt(n)
        # Increment by 2 to skip even numbers
        step = 2
        i = 3
        while i <= limit:
            if n % i == 0:
                return i
            i += step
        return None

    def solve(self, problem: Dict[str, int]) -> Dict[str, int]:
        """
        Find the two prime factors `p` and `q` of the composite integer.

        Parameters
        ----------
        problem : dict
            Must contain a single key ``'composite'`` whose value is an int.

        Returns
        -------
        dict
            ``{'p': p, 'q': q}`` where `p < q` and both are prime.

        Raises
        ------
        ValueError
            If the input is not a valid integer,
            or if the integer does not have exactly two prime factors.
        """
        if not isinstance(problem, dict):
            raise ValueError(f"Expected a dict, got {type(problem)}")

        composite_val = problem.get("composite")
        if not isinstance(composite_val, int):
            raise ValueError(f"Composite value must be an int, got {composite_val!r}")

        # Find the first factor
        f1 = self._prime_factor(composite_val)
        if f1 is None or f1 == composite_val:
            raise ValueError("Composite number is prime or 1, not a product of two primes")

        f2 = composite_val // f1

        # Verify both factors are prime
        if not (sympy.isprime(f1) and sympy.isprime(f2)):
            raise ValueError("Found factors are not both prime")

        p, q = sorted((int(f1), int(f2)))
        return {"p": p, "q": q}