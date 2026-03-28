import random
from typing import Any, Dict
from math import gcd, isqrt

# A lightweight and fast Pollard Rho implementation for 64‑bit integers.
# The algorithm is deterministic for typical contest sizes (up to ~10^18).

def _pollard_rho(n: int) -> int:
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
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

def _factor(n: int, res: list[int]) -> None:
    if n == 1:
        return
    if _is_prime(n):
        res.append(n)
        return
    d = _pollard_rho(n)
    _factor(d, res)
    _factor(n // d, res)

def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    small_primes = [2,3,5,7,11,13,17,19,23,29,31,37]
    for p in small_primes:
        if n % p == 0:
            return n == p
    d, s = n-1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    # Deterministic Miller-Rabin bases for 64‑bit integers
    for a in [2,325,9375,28178,450775,9780504,1795265022]:
        if a % n == 0:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n-1:
            continue
        for _ in range(s-1):
            x = pow(x, 2, n)
            if x == n-1:
                break
        else:
            return False
    return True

class Solver:
    def solve(self, problem: Dict[str, int]) -> Dict[str, int]:
        composite_val = problem['composite']
        n = int(composite_val)
        if n <= 1:
            raise ValueError(f"Composite must be >1, got {n}")
        factors: list[int] = []
        _factor(n, factors)
        if len(factors) != 2:
            raise ValueError(f"Expected exactly two prime factors, got {len(factors)}")
        p, q = sorted(factors)
        return {'p': p, 'q': q}