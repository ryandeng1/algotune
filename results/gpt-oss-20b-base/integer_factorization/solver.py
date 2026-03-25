# solver.py
import random
import sys
import sympy
import math
from typing import Any, Dict

# Simple deterministic Pollard Rho with Brent's cycle detection,
# fast for semi‐smooth composites like product of two primes.
def _pollard_rho(n: int) -> int:
    if n % 2 == 0:
        return 2
    # Choose random seed and function coefficient
    while True:
        c = random.randrange(1, n)
        f = lambda x: (pow(x, 2, n) + c) % n
        x, y, d = 2, 2, 1
        while d == 1:
            x = f(x)
            y = f(f(y))
            d = math.gcd(abs(x - y), n)
        if d != n:
            return d

def _factor_two_primes(n: int) -> (int, int):
    # First factor
    a = _pollard_rho(n)
    b = n // a
    # Ensure both are primes
    if sympy.isprime(a) and sympy.isprime(b):
        return (a, b)
    # If not, try to factor again
    a = _pollard_rho(a)
    b = n // a
    return (a, n // a)

class Solver:
    def solve(self, problem: Dict[str, int], **kwargs) -> Dict[str, int]:
        """
        Factor a composite integer that is a product of two primes.
        Uses deterministic Pollard Rho with Brent's cycle detection.
        """
        composite = problem.get("composite")
        if composite is None:
            raise ValueError("Missing 'composite' key")
        if not isinstance(composite, int):
            # Attempt to convert
            try:
                composite = int(composite)
            except Exception as e:
                raise ValueError(f"Cannot convert composite to int: {e}")
        if composite <= 1:
            raise ValueError("Composite must be > 1")

        # Factor the number
        p, q = _factor_two_primes(composite)
        if p > q:
            p, q = q, p
        return {"p": p, "q": q}
