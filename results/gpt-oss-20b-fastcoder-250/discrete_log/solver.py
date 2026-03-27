import math
from typing import Dict

def solve(problem: Dict[str, int]) -> Dict[str, int]:
    """
    Compute the discrete logarithm x such that g^x ≡ h (mod p)
    using an efficient baby‑step giant‑step algorithm.
    The algorithm runs in O(√p) time and O(√p) space.

    Parameters
    ----------
    problem : dict
        Dictionary containing the keys ``p``, ``g`` and ``h``.
        ``p`` must be prime, ``g`` its primitive root and ``h`` the target element.

    Returns
    -------
    dict
        The dictionary {"x": solution}. If no solution exists,
        ``x`` will be ``None``.
    """
    p = problem["p"]
    g = problem["g"]
    h = problem["h"]

    # Baby‑step–giant‑step
    N = p - 1                # order of the group
    m = math.isqrt(N) + 1    # ceil(sqrt(N))

    # Baby steps: g^j mod p for j = 0 .. m-1
    baby = {}
    e = 1
    for j in range(m):
        if e not in baby:          # keep the smallest exponent
            baby[e] = j
        e = (e * g) % p

    # Pre‑compute g^{-m} mod p
    inv_g_m = pow(g, -m, p)

    # Giant steps
    gamma = h
    for i in range(m):
        if gamma in baby:
            return {"x": i * m + baby[gamma]}      # solution found
        gamma = (gamma * inv_g_m) % p

    # No solution
    return {"x": None}