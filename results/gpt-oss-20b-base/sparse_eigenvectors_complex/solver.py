import numpy as np
import scipy.sparse.linalg

def solve(problem: dict) -> list[complex]:
    """
    Solve the eigenvalue problem for the given square sparse matrix.
    The solution returned is a list of eigenvectors with the largest `k`
    eigenvalues sorted in descending order by their modulus.

    :param problem: A dictionary containing the sparse matrix under the key
                    'matrix' and the number of eigenvalues `k` under the key
                    'k'.
    :return: List of eigenvectors sorted in descending order by the modulus
             of the corresponding eigenvalue.
    """
    A = problem["matrix"]
    k = problem["k"]

    # Use the default random vector to seed the Arnoldi iteration – this
    # is usually faster than creating a large one‑vector and works with any
    # matrix type.
    evals, evecs = scipy.sparse.linalg.eigs(
        A, k=k, maxiter=A.shape[0] * 200, ncv=max(2 * k + 1, 20)
    )

    # Sort indices by descending modulus of the eigenvalues
    idx = np.argsort(-np.abs(evals))

    # Return the eigenvectors in sorted order
    return [evecs[:, i] for i in idx]