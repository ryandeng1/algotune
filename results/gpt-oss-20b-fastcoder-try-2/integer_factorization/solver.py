import random
import math
from typing import Dict, List

# Helper: fast modular exponentiation
def _pow_mod(base: int, exp: int, mod: int) -> int:
    result = 1
    base %= mod
    while exp:
        if exp & 1:
            result = (result * base) % mod
        base = (base * base) % mod
        exp >>= 1
    return result

# Miller‑Rabin deterministic for 64‑bit integers
def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    # bases sufficient for n < 2^64
    for a in (2, 325, 9375, 28178, 450775, 9780504, 1795265022):
        if a % n == 0:
            continue
        x = _pow_mod(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = (x * x) % n
            if x == n - 1:
                break
        else:
            return False
    return True

# Pollard‑Rho factorization
def _pollard_rho(n: int) -> int:
    if n % 2 == 0:
        return 2
    while True:
        c = random.randrange(1, n)
        f = lambda x: (x * x + c) % n
        x, y, d = 2, 2, 1
        while d == 1:
            x = f(x)
            y = f(f(y))
            d = math.gcd(abs(x - y), n)
        if d != n:
            return d

def _factor(n: int) -> List[int]:
    if n == 1:
        return []
    if _is_prime(n):
        return [n]
    d = _pollard_rho(n)
    return _factor(d) + _factor(n // d)

class Solver:
    def solve(self, problem: Dict[str, int]) -> Dict[str, int]:
        """
        Factor the composite number into two primes using Pollard‑Rho
        and deterministic Miller‑Rabin for primality testing.

        :param problem: dict containing key 'composite' with the number to factor.
        :return: dict with keys 'p' and 'q', where p < q.
        :raises ValueError: if the number doesn't factor into exactly two primes.
        """
        n = problem['composite']
        if n <= 1 or not isinstance(n, int):
            raise ValueError(f"Invalid composite number: {n}")

        factors = sorted(_factor(n))
        if len(factors) != 2:
            raise ValueError(f'Expected 2 factors, but got {len(factors)}.')
        return {'p': factors[0], 'q': factors[1]}