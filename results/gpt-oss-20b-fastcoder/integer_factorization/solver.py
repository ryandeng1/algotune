import random
import math
from typing import Dict

# ------------------------------------------------------------
# Miller‑Rabin primality test (deterministic for 64‑bit ints)
# ------------------------------------------------------------
def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for p in small_primes:
        if n % p == 0:
            return n == p
    # write n-1 as d*2^s
    d, s = n - 1, 0
    while d & 1 == 0:
        d >>= 1
        s += 1
    # deterministically safe bases for n < 2**64
    for a in [2, 325, 9375, 28178, 450775, 9780504, 1795265022]:
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

# ------------------------------------------------------------
# Pollard‑Rho factorisation (recursive, returns *all* prime factors)
# ------------------------------------------------------------
def _pollard_rho(n: int) -> int:
    if n % 2 == 0:
        return 2
    # choose random polynomial f(x) = x^2 + c
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

def _factor(n: int, res: list):
    if n == 1:
        return
    if _is_prime(n):
        res.append(n)
        return
    d = _pollard_rho(n)
    _factor(d, res)
    _factor(n // d, res)

# ------------------------------------------------------------
# Solver
# ------------------------------------------------------------
class Solver:
    def solve(self, problem: Dict[str, int]) -> Dict[str, int]:
        """
        Factorises a composite number into exactly two prime factors
        using a fast Pollard‑Rho algorithm.

        :param problem: Dictionary containing the composite integer under key 'composite'.
        :return: Dictionary with keys 'p' and 'q' containing the two prime factors (p < q).
        :raises ValueError: If the factorisation does not yield exactly two primes.
        """
        n = int(problem['composite'])
        if n <= 1:
            raise ValueError(f"Input must be a composite integer > 1, got {n}")
        factors: list[int] = []
        _factor(n, factors)
        if len(factors) != 2:
            raise ValueError(f'Expected 2 factors, but got {len(factors)}.')
        p, q = sorted(map(int, factors))
        return {'p': p, 'q': q}