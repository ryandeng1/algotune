import random
from math import gcd, isqrt
from typing import Dict

# Miller–Rabin primality test deterministic for 64‑bit integers
_SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

def _is_prime(n: int) -> bool:
    """Deterministic Miller‑Rabin for 64‑bit integers."""
    if n < 2:
        return False
    for p in _SMALL_PRIMES:
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    # Bases sufficient for all 64‑bit integers
    for a in [2, 325, 9375, 28178, 450775, 9780504, 1795265022]:
        if a % n == 0:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = (x * x) % n
            if x == n - 1:
                break
        else:
            return False
    return True

def _pollard_rho(n: int) -> int:
    """Return a non‑trivial factor of n using Pollard's Rho."""
    if n % 2 == 0:
        return 2
    while True:
        c = random.randrange(1, n)
        f = lambda x: (pow(x, 2, n) + c) % n
        x, y, d = 2, 2, 1
        while d == 1:
            x = f(x)
            y = f(f(y))
            d = gcd(abs(x - y), n)
        if d != n:
            return d

def _factor_two_primes(n: int) -> tuple[int, int]:
    """Assumes n is a product of exactly two primes."""
    # Quick check for square
    r = isqrt(n)
    if r * r == n:
        return r, r
    # Find one factor
    f = _pollard_rho(n)
    # Ensure f is prime; otherwise factor further
    if not _is_prime(f):
        f = _factor_two_primes(f)[0]
    q = n // f
    if not _is_prime(q):
        q = _factor_two_primes(q)[0]
    return min(f, q), max(f, q)

class Solver:
    def solve(self, problem: Dict[str, int]) -> Dict[str, int]:
        composite_val = problem["composite"]
        if not isinstance(composite_val, int):
            raise ValueError(f"Composite value '{composite_val}' is not an integer.")
        if composite_val <= 1:
            raise ValueError("Composite must be > 1.")
        p, q = _factor_two_primes(composite_val)
        if not (_is_prime(p) and _is_prime(q)):
            raise ValueError(f"Failed to factor into two primes: got {p} and {q}.")
        return {"p": p, "q": q}