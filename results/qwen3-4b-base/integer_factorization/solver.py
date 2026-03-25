import math
import random

class Solver:
    def solve(self, problem: dict[str, int]) -> dict[str, int]:
        composite = problem["composite"]
        if composite % 2 == 0:
            return {"p": 2, "q": composite // 2}
        d = self.pollard_rho(composite)
        p = d
        q = composite // d
        if p > q:
            p, q = q, p
        return {"p": p, "q": q}
    
    def pollard_rho(self, n):
        if n % 2 == 0:
            return 2
        x = random.randrange(2, n - 1)
        y = x
        c = random.randrange(1, n - 1)
        d = 1
        while d == 1:
            x = (x * x + c) % n
            y = (y * y + c) % n
            y = (y * y + c) % n
            d = math.gcd(abs(x - y), n)
            if d == n:
                return self.pollard_rho(n)
        return d
