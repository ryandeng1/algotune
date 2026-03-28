def solve(self, problem: list[float]) -> list[complex]:
    coefficients = problem
    with _single_thread_blas():
        computed_roots = np.roots(coefficients)
    key = np.column_stack((-computed_roots.real, -computed_roots.imag))
    sorted_indices = np.argsort(key)
    return computed_roots[sorted_indices].tolist()