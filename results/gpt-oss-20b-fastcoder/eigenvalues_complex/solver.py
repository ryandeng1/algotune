import numpy as np

class Solver:
    def solve(self, problem: np.ndarray) -> list[complex]:
        """
        Solve the eigenvalue problem for the given square matrix.
        The solution returned is a list of eigenvalues sorted in descending order.
        The sorting order is defined as follows: first by the real part (descending),
        then by the imaginary part (descending).

        :param problem: A numpy array representing the real square matrix.
        :return: List of eigenvalues (complex numbers) sorted in descending order.
        """
        # Fast eigenvalue computation
        eigs = np.linalg.eigvals(problem)

        # Lexicographical sort: first by real part descending, then by imag part descending
        # np.lexsort sorts according to the last key first, so we provide keys in reverse order
        order = np.lexsort((-eigs.imag, -eigs.real))
        # Reorder and convert to plain Python list
        return list(eigs[order])