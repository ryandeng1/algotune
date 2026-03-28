from math import isqrt

class Solver:
    def solve(self, problem: dict[str, int]) -> dict[str, int]:
        """
        Compute the discrete logarithm x such that g^x ≡ h (mod p)
        using the Baby‑Step Giant‑Step algorithm.

        :param problem: Dictionary with keys 'p', 'g', 'h'.
        :return: Dictionary with key 'x' containing the solution,
                 or -1 if no solution exists.
        """
        p = problem["p"]
        g = problem["g"]
        h = problem["h"]

        # The search space is [0, p-2] because g is a generator of Z*_p.
        # Use baby‑step giant‑step with step size m.
        m = isqrt(p - 2) + 1

        # Baby steps: g^j for j in [0, m)
        table = {}
        cur = 1
        for j in range(m):
            if cur not in table:          # keep smallest j
                table[cur] = j
            cur = (cur * g) % p

        # Compute g^(-m) mod p using Fermat's little theorem
        factor = pow(g, p - 1 - m, p)

        gamma = h
        for i in range(m):
            if gamma in table:
                return {"x": i * m + table[gamma]}
            gamma = (gamma * factor) % p

        # No solution found
        return {"x": -1}