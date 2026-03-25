import numba
from numba import jit
import math

@jit(nopython=True)
def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

@jit(nopython=True)
def pollard_rho(n):
    if n % 2 == 0:
        return 2
    x = 2
    y = 2
    d = 1
    c = 1
    while d == 1:
        x = (x * x + c) % n
        y = (y * y + c) % n
        y = (y * y + c) % n
        d = gcd(abs(x - y), n)
        if d == n:
            c += 1
            x = 2
            y = 2
            d = 1
    return d

class Solver:
    def solve(self, problem, **kwargs) -> dict:
        composite = problem["composite"]
        if composite % 2 == 0:
            return {"p": 2, "q": composite // 2}
        p = pollard_rho(composite)
        q = composite // p
        if p > q:
            p, q = q, p
        return {"p": p, "q": q}
