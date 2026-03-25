import math

class Solver:
    def solve(self, problem: dict[str, int]) -> dict[str, int]:
        p = problem["p"]
        g = problem["g"]
        h = problem["h"]
        
        # Handle trivial cases
        if h == 1:
            return {"x": 0}
        if p == 2:
            return {"x": 0}
        
        n = p - 1
        m = math.isqrt(n) + 1
        
        # Baby steps: g^j mod p for j in 0..m-1
        baby_steps = {}
        current = 1
        for j in range(m):
            baby_steps[current] = j
            current = (current * g) % p
        
        # Giant steps
        g_m = pow(g, m, p)
        g_m_inv = pow(g_m, p - 2, p)
        current = h
        for i in range(m):
            if current in baby_steps:
                j = baby_steps[current]
                return {"x": i * m + j}
            current = (current * g_m_inv) % p
        
        return {"x": 0}
