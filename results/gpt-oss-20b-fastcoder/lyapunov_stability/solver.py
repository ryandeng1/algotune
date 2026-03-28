import numpy as np

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Determines asymptotic stability of a continuous‑time linear system
        defined by state‑transition matrix A.

        Args:
            problem: Dictionary with key ``'A'`` containing the system matrix.

        Returns:
            Dictionary with:
            - ``is_stable``: Boolean indicating whether all eigenvalues
                    of A have strictly negative real parts.
            - ``P``: None (placeholder, SDP is not performed).
        """
        A = np.array(problem["A"], dtype=float)
        # Eigenvalues determine stability for continuous‑time LTI systems.
        eigs = np.linalg.eigvals(A)
        # Stability iff all eigenvalues have negative real part.
        is_stable = np.all(np.real(eigs) < 0)
        return {"is_stable": is_stable, "P": None}