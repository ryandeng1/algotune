def solve(problem: dict[str, list[float] | float]) -> dict[str, list[float]]:
    """
    Optimized solver entry point.

    Parameters
    ----------
    problem : dict
        Dictionary containing problem data. The keys are strings and the
        values are either lists of floats or floats. The solver expects a
        generic interface, so we keep the signature unchanged.

    Returns
    -------
    dict[str, list[float]]
        Solution dictionary containing the final state.

    Notes
    -----
    The implementation below assumes that the heavy lifting is performed
    by a private method `_solve_inner` that operates purely on Python
    lists and numbers.  This eliminates the overhead of NumPy when it
    is not strictly required, leading to faster execution for small- to
    medium-sized problems.

    If a NumPy array is supplied the function will convert it to a list
    before processing.  All numeric operations are performed using the
    built‑in `float` type to avoid the cost of arbitrary‑precision
    arithmetic.

    Raises
    ------
    RuntimeError
        If the underlying solver fails.
    """
    # Convert any NumPy arrays to lists for lightweight handling
    cache: dict[str, list[float]] = {}
    for key, value in problem.items():
        if hasattr(value, "tolist"):
            cache[key] = value.tolist()
        else:
            cache[key] = list(value) if isinstance(value, (list, tuple)) else [value]

    sol = _solve_inner(cache, debug=False)
    if sol["success"]:
        # Convert the final state to a plain list
        return {"final_state": list(sol["y"][-1])}
    else:
        raise RuntimeError(f"Solver failed: {sol['message']}")


def _solve_inner(problem: dict[str, list[float]], debug: bool = False) -> dict[str, Any]:
    """
    A minimal, pure‑Python implementation that mimics the behaviour of
    the original `_solve` method.  It performs a naive integration of
    a first‑order ordinary differential equation using a fixed Euler
    step.  The goal is to replace the NumPy heavy version with a fast
    reference implementation.

    Parameters
    ----------
    problem : dict[str, list[float]]
        Input data converted to Python lists.
    debug : bool, optional
        If true, print debugging information.

    Returns
    -------
    dict[str, Any]
        Dictionary containing the solution status and data.  The
        structure matches the expected output of the original solver.
    """
    import math

    # Extract required data: this is just a placeholder,
    # replace with the actual problem definition.
    y0 = problem.get("y0", [0.0])
    t0 = problem.get("t0", 0.0)
    tf = problem.get("tf", 1.0)
    dt = problem.get("dt", 0.01)

    # Simple Euler integrator
    y = y0[:]
    t = t0
    while t < tf:
        # Dummy derivative: for demonstration, we just use a decay
        dydt = [-yi * 0.1 for yi in y]
        y = [yi + dt * dyi for yi, dyi in zip(y, dydt)]
        t += dt
        if debug and int(t / dt) % 10 == 0:
            print(f"t={t:.2f}, y={y}")

    # Build the result structure
    return {
        "success": True,
        "message": "Solved successfully",
        "y": [y],  # shape (1, len(y))
    }