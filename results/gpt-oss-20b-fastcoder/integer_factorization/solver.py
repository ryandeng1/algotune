def solve(problem: dict[str, int]) -> dict[str, int]:
    """
    Very fast factorization of a composite integer that has exactly two prime factors.
    Works for 64‑bit unsigned integers (i.e. up to < 2**63) which is more than
    sufficient for typical RSA challenge inputs.
    """
    n = problem["composite"]

    # ----- helpers -----------------------------------------------------------

    def _is_probable_prime(x: int) -> bool:
        """Deterministic Miller–Rabin for 64‑bit integers."""
        if x < 2:
            return False
        # small primes
        small = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
        for p in small:
            if x % p == 0:
                return x == p
        # write x-1 as d*2**s
        d, s = x - 1, 0
        while d & 1 == 0:
            d >>= 1
            s += 1
        # bases that are sufficient for 64‑bit integers
        for a in (2, 325, 9375, 28178, 450775, 9780504, 1795265022):
            if a % x == 0:
                continue
            ac = pow(a, d, x)
            if ac == 1 or ac == x - 1:
                continue
            for _ in range(s - 1):
                ac = (ac * ac) % x
                if ac == x - 1:
                    break
            else:
                return False
        return True

    # ----- factorization ----------------------------------------------------

    # handle trivial even factor
    if n % 2 == 0:
        p = 2
        q = n // 2
        if _is_probable_prime(q):
            return {"p": p, "q": q}

    # trial division by odd numbers up to sqrt(n)
    # use increments of 6 (6k±1) for speed
    i = 3
    limit = int(n**0.5) + 1
    while i <= limit and i <= 50_000:   # skip huge loops if impossible
        if n % i == 0:
            p, q = i, n // i
            if _is_probable_prime(p) and _is_probable_prime(q):
                return {"p": min(p, q), "q": max(p, q)}
        i += 2

    # if we reached here, n is either prime or product of two large primes,
    # thus we need a more robust method: Pollard’s Rho.
    # We use a deterministic variant that works for 64-bit numbers.

    def _rho(n: int) -> int:
        if n % 2 == 0:
            return 2
        import random, math
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

    factor = _rho(n)
    p, q = factor, n // factor
    p, q = min(p, q), max(p, q)
    if not (_is_probable_prime(p) and _is_probable_prime(q)):
        raise ValueError("Failed to factor the composite correctly.")
    return {"p": p, "q": q}