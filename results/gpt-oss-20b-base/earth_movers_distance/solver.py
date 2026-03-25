import numpy as np
from scipy.optimize import linprog


class Solver:
    def solve(self, problem, **kwargs):
        """
        Solve the Earth Mover's Distance (optimal transport) problem using
        the high‑performance linear programming solver from SciPy.

        Parameters
        ----------
        problem : dict[str, np.ndarray]
            Must contain the keys:
                - "source_weights": 1‑D array-like, sums to 1.0
                - "target_weights": 1‑D array-like, sums to the same total mass
                - "cost_matrix": 2‑D array-like of shape (len(source_weights), len(target_weights))

        Returns
        -------
        dict
            A dictionary with a single key "transport_plan" whose value is a
            list of lists representing the optimal transport plan matrix G.
        """
        # Extract and convert to NumPy arrays
        a = np.asarray(problem["source_weights"], dtype=np.float64).ravel()
        b = np.asarray(problem["target_weights"], dtype=np.float64).ravel()
        M = np.asarray(problem["cost_matrix"], dtype=np.float64)

        n = a.size
        m = b.size
        # Linear programming formulation:
        # variables: x[i*m + j] = G[i, j]
        # objective: c = M.flatten()
        c = M.ravel()

        # Constraints:
        # 1. Row sums equal a (n constraints)
        # 2. Column sums equal b (m constraints)
        # 3. Non‑negativity automatically enforced by linprog

        # Build equality constraint matrix A_eq and vector b_eq
        # Row constraints: each row sum -> 1 at positions of that row
        A_eq_rows = np.zeros((n, n * m))
        for i in range(n):
            A_eq_rows[i, i * m : (i + 1) * m] = 1.0

        # Column constraints: each column sum -> 1 at positions of that column
        A_eq_cols = np.zeros((m, n * m))
        for j in range(m):
            A_eq_cols[j, j::m] = 1.0

        A_eq = np.vstack((A_eq_rows, A_eq_cols))
        b_eq = np.concatenate((a, b))

        # Solve LP using HiGHS (default fast solver in SciPy 1.10+)
        res = linprog(
            c,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=(0, None),
            method="highs",
            options={"presolve": True},
        )

        if not res.success:
            # Fallback to SciPy's older FUNAB method if HiGHS fails
            res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs-ds")
            if not res.success:
                raise RuntimeError(f"Linear program failed: {res.message}")

        G = res.x.reshape((n, m))
        # Convert to Python list of lists for the expected output format
        transport_plan = G.tolist()

        return {"transport_plan": transport_plan}
