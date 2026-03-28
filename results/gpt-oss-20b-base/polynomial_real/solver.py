import numpy as np

def solve(problem: list[float]) -> list[float]:
    """
    Find all real roots of a polynomial given by its coefficients
    in descending order.

    Parameters
    ----------
    problem : list[float]
        Coefficients of the polynomial [aₙ, aₙ₋₁, …, a₀].

    Returns
    -------
    list[float]
        Real roots sorted in descending order.
    """
    # Convert to ndarray once (avoid repeated conversions)
    coeffs = np.asarray(problem, dtype=float)

    # Compute all roots (complex in general)
    roots = np.roots(coeffs)

    # Discard negligible imaginary parts
    # `real_if_close` keeps the value as complex if the imag part is
    # > tol; otherwise it casts to real.  The default tol works well for
    # typical numerical noise.
    roots = np.real_if_close(roots, tol=0.001)

    # Extract only the real part (now guaranteed to be real)
    roots = np.asarray(roots.real, dtype=float)

    # Sort in descending order
    roots.sort()
    return roots[::-1].tolist()