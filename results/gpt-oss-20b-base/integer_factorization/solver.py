import sys
from typing import Dict
import random

def _is_probable_prime(n: int) -> bool:
    if n < 2:
        return False
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
    for p in small_primes:
        if n % p == 0:
            return n == p
    # Miller–Rabin deterministic for 64‑bit integers
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
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

def _pollard_rho(n: int) -> int:
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

def _factor_two_primes(n: int) -> tuple[int, int]:
    if _is_probable_prime(n):
        raise ValueError("The input is itself prime")
    d = _pollard_rho(n)
    a, b = n // d, d
    if not _is_probable_prime(a) or not _is_probable_prime(b):
        # One factor may still be composite; keep factorising
        facs = []
        for f in (a, b):
            if _is_probable_prime(f):
                facs.append(f)
            else:
                facs.extend(_factor_two_primes(f))
        if len(facs) != 2:
            raise ValueError(f"Expected 2 prime factors, got {len(facs)}")
        return tuple(sorted(facs))
    return tuple(sorted((a, b)))

class Solver:
    def solve(self, problem: Dict[str, int]) -> Dict[str, int]:
        composite = problem["composite"]
        p, q = _factor_two_primes(int(composite))
        return {"p": p, "q": q}