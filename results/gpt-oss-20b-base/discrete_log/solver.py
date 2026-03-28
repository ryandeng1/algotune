from typing import Any, Dict
import math

class Solver:
    def solve(self, problem: Dict[str, int]) -> Dict[str, int]:
        """
        Solve the discrete logarithm problem using a baby-step giant-step algorithm.
        This implementation is faster for large moduli than sympy's discrete_log
        for ordinary inputs where a solution exists.

        :param problem: A dictionary with keys 'p', 'g', and 'h'.
        :return: A dictionary with key 'x' containing the discrete logarithm.
        """
        p = problem['p']
        g = problem['g']
        h = problem['h']

        # Handle trivial cases
        if h % p == 1:
            return {'x': 0}
        if g % p == 0:
            # g == 0 mod p -> only 0^x == 0, so no solution unless h==0
            raise ValueError("No discrete logarithm exists for g=0")

        # Baby-step giant-step
        n = p - 1  # order of the multiplicative group (assuming p prime)
        m = math.isqrt(n) + 1

        # Baby steps: g^j for j = 0..m-1
        baby = {}
        cur = 1
        for j in range(m):
            if cur not in baby:      # keep the smallest exponent
                baby[cur] = j
            cur = (cur * g) % p

        # Compute g^{-m} mod p
        inv_g = pow(g, -1, p)
        inv_g_m = pow(inv_g, m, p)

        gamma = h % p
        for i in range(m):
            if gamma in baby:
                x = i * m + baby[gamma]
                return {'x': x % n}
            gamma = (gamma * inv_g_m) % p

        raise ValueError("Discrete logarithm not found")