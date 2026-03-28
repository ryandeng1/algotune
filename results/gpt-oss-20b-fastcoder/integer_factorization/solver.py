import random
from typing import Dict

# Miller–Rabin primality test, deterministic for n < 2^64
def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    # bases for 64‑bit integers
    for a in (2, 325, 9375, 28178, 450775, 9780504, 1795265022):
        if a % n == 0:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True

def pollard_rho(n: int) -> int:
    if n % 2 == 0:
        return 2
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

import math

def factor(n: int) -> list[int]:
    """Return the prime factors of n (repeated)."""
    if n <= 1:
        return []
    if is_prime(n):
        return [n]
    d = pollard_rho(n)
    return sorted(factor(d) + factor(n // d))

class Solver:
    def solve(self, problem: Dict[str, int]) -> Dict[str, int]:
        composite = problem['composite']
        if not isinstance(composite, int):
            raise ValueError(f"Composite must be an int, got {type(composite)}")
        factors = factor(composite)
        if len(factors) != 2:
            raise ValueError(f"Expected 2 factors, got {len(factors)}")
        p, q = sorted(factors)
        return {'p': p, 'q': q}