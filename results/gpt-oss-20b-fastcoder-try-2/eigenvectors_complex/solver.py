import numpy as np

class Solver:
    def solve(self, problem: np.ndarray):
        """
        Compute eigenvalues/eigenvectors of a non‑symmetric matrix.
        Sort the eigenpairs by (−Reλ, −Imλ) and return normalized eigenvectors
        as a list of lists of complex numbers.
        """
        # Compute eigenpairs
        eigvals, eigvecs = np.linalg.eig(problem)

        # --- sorting -------------------------------------------------------
        # Get sorting indices (descending by real part, then imaginary part)
        # Rearrange into a structured array for stable sorting
        dtype = np.dtype([('real', eigvals.dtype), ('imag', eigvals.dtype), ('idx', np.int32)])
        structured = np.array(list(zip(eigvals.real, eigvals.imag, np.arange(len(eigvals)))),
                              dtype=dtype)
        # Sort descending by real, then imaginary
        order = np.argsort(structured, order=('real', 'imag'))[::-1]
        sorted_idx = structured['idx'][order]

        # Reorder eigenvectors
        eigvecs = eigvecs[:, sorted_idx]

        # --- normalization ------------------------------------------------
        # Compute norms of each eigenvector (column)
        norms = np.linalg.norm(eigvecs, axis=0)
        # Avoid division by zero – treat zero norms as 1 (will keep vector zero)
        inv_norms = np.where(norms > 1e-12, 1.0 / norms, 0.0)
        # Normalize columns
        eigvecs = eigvecs * inv_norms

        # Convert to list of lists
        return eigvecs.T.tolist()