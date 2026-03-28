def solve(problem: dict[str, list[float]]) -> float:
    """
    Compute the 1‑D Wasserstein distance (earth mover's distance) for two
    probability distributions u and v defined on the integer grid 1..n.
    The formula for equally spaced points is the sum of absolute values
    of the cumulative difference between the two distributions.

    :param problem: dict with keys 'u' and 'v' containing the two
                    distributions as lists of floats.
    :return: the Wasserstein distance (float).
    """
    u = problem.get('u')
    v = problem.get('v')
    if u is None or v is None:
        return float('nan')

    # Both distributions must have the same length
    n = len(u)
    if n != len(v) or n == 0:
        return float('nan')

    cum_diff = 0.0
    distance = 0.0
    for ui, vi in zip(u, v):
        cum_diff += ui - vi
        distance += abs(cum_diff)

    return distance