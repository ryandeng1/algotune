from typing import Any
import numpy as np
import cvxpy as cp


class Solver:
    def solve(self, problem: dict) -> dict:
        """Solve a tensor completion problem by nuclear‑norm minimisation of
        the three mode‑unfoldings.  The implementation avoids explicit Python
        loops and uses NumPy broadcasting for efficiency."""
        # ------------------------------------------------------------------
        # 1. Extract data
        # ------------------------------------------------------------------
        tensor = np.array(problem["tensor"], dtype=np.float64)
        mask   = np.array(problem["mask"],   dtype=bool)
        d1, d2, d3 = tensor.shape

        # ------------------------------------------------------------------
        # 2. Unfold the tensor and the mask along each mode
        # ------------------------------------------------------------------
        # mode 1 : (d1) × (d2*d3)
        U1   = tensor.reshape(d1, d2 * d3)
        M1   = mask.reshape(d1, d2 * d3)

        # mode 2 : (d2) × (d1*d3)  – use transpose to avoid loops
        U2   = tensor.transpose(1, 0, 2).reshape(d2, d1 * d3)
        M2   = mask.transpose(1, 0, 2).reshape(d2, d1 * d3)

        # mode 3 : (d3) × (d1*d2)  – use transpose to avoid loops
        U3   = tensor.transpose(2, 0, 1).reshape(d3, d1 * d2)
        M3   = mask.transpose(2, 0, 1).reshape(d3, d1 * d2)

        # ------------------------------------------------------------------
        # 3. CVXPY optimisation variables
        # ------------------------------------------------------------------
        X1 = cp.Variable((d1, d2 * d3))
        X2 = cp.Variable((d2, d1 * d3))
        X3 = cp.Variable((d3, d1 * d2))

        # ------------------------------------------------------------------
        # 4. Objective: sum of nuclear norms
        # ------------------------------------------------------------------
        objective = cp.Minimize(cp.norm(X1, "nuc") + cp.norm(X2, "nuc") + cp.norm(X3, "nuc"))

        # ------------------------------------------------------------------
        # 5. Data‑fidelity constraints (only on observed entries)
        # ------------------------------------------------------------------
        constraints = [
            cp.multiply(X1, M1) == cp.multiply(U1, M1),
            cp.multiply(X2, M2) == cp.multiply(U2, M2),
            cp.multiply(X3, M3) == cp.multiply(U3, M3),
        ]

        # ------------------------------------------------------------------
        # 6. Solve the problem
        # ------------------------------------------------------------------
        prob = cp.Problem(objective, constraints)
        try:
            # Use a fast solver if available
            prob.solve(solver=cp.SCS, eps=1e-6, max_iters=4000)

            if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or X1.value is None:
                return {"completed_tensor": []}

            # Re‑fold the solution from the first mode to obtain the tensor
            completed = X1.value.reshape(tensor.shape)
            return {"completed_tensor": completed.tolist()}

        except cp.SolverError:
            return {"completed_tensor": []}
        except Exception:
            return {"completed_tensor": []}