class Solver:
    def solve(self, problem: dict[str, int]) -> dict[str, int]:
        p = problem["p"]
        g = problem["g"]
        h = problem["h"]
        
        # Handle trivial cases
        if h == 1:
            return {"x": 0}
        if g == 1:
            return {"x": 0}
        if g == h:
            return {"x": 1}
        
        m = int(p**0.5) + 1
        
        baby_steps = {}
        current = 1
        for i in range(m):
            if current not in baby_steps:
                baby_steps[current] = i
            current = (current * g) % p
        
        inv_gm = pow(g, -m, p)
        current = h
        for j in range(m):
            if current in baby_steps:
                return {"x": baby_steps[current] + j * m}
            current = (current * inv_gm) % p
        
        return {"x": -1}