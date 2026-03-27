from math import isqrt
from typing import Dict

def solve(problem: Dict[str, int]) -> Dict[str, int]:
    """Solve the discrete logarithm g**x = h (mod p) using baby‑step giant‑step.

    The algorithm runs in O(sqrt(p) * log p) time with O(sqrt(p)) memory.
    """
    p: int = problem["p"]
    g: int = problem["g"] % p
    h: int = problem["h"] % p

    # Trivial cases
    if h == 1:
        return {"x": 0}
    if g == 0:
        return {"x": 1 if h == 0 else None}

    # Pre‑compute baby steps: g^j (mod p) for j = 0 .. m-1
    m = isqrt(p) + 1
    baby = {}
    cur = 1
    for j in range(m):
        if cur not in baby:        # keep smallest exponent
            baby[cur] = j
        cur = (cur * g) % p

    # Compute g^-m mod p
    inv_g_m = pow(g, (p - 1 - m) % (p - 1), p)

    # Giant steps: h * g^(-i*m) (mod p)
    cur = h
    for i in range(m):
        if cur in baby:
            return {"x": i * m + baby[cur]}
        cur = (cur * inv_g_m) % p

    # No solution found
    return {"x": None}