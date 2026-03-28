from typing import Dict
import random
import math

def _is_probable_prime(n: int) -> bool:
    if n < 2:
        return False
    # small primes
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    for p in small_primes:
        if n % p == 0:
            return n == p
    # Miller-Rabin
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    # deterministic bases for 64‑bit numbers
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

def _factor(n: int) -> list[int]:
    if n == 1:
        return []
    if _is_probable_prime(n):
        return [n]
    d = _pollard_rho(n)
    return _factor(d) + _factor(n // d)

def solve(problem: Dict[str, int]) -> Dict[str, int]:
    """
    Factor the provided composite into exactly two prime factors.
    The result will contain the smaller prime under key 'p' and the larger under key 'q'.
    """
    composite_val = problem['composite']
    # ensure integer
    if not isinstance(composite_val, int):
        raise ValueError(f"Composite value must be int, got {type(composite_val)}")
    factors = _factor(composite_val)
    if len(factors) != 2:
        raise ValueError(f'Expected 2 factors, but got {len(factors)}.')
    p, q = sorted(map(int, factors))
    return {'p': p, 'q': q}