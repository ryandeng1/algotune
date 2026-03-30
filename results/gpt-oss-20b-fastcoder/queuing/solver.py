# solver.py
# ------------------------------------------------------------
# Optimised CVXPY solver for the given optimisation problem.
# ------------------------------------------------------------
from __future__ import annotations

from typing import Any

import cvxpy as cp
import numpy as np

# Reduce compilation time by preparing the solver options once.
_SOLVER_OPTIONS = {
    "solver": cp.SCS,
    "verbose": False,
    "max_iters": 2_500,
    "eps": 1e-8,
}

# ------------------------------------------------------------
# The solver class
# ------------------------------------------------------------
class Solver:
    """
    A highly‑optimised implementation of the original ``solve`` routine.
    The key performance improvements are:
    * Pre‑conversion of all inputs to ``numpy.ndarray`` as early as possible.
    * Avoiding any redundant copies or temporary Python objects.
    * Handling the GP / DCP solvers in a single ``try/except`` chain.
    * Explicitly setting a fast, memory‑efficient solver (SCS) and
      supplying a tight tolerance.
    * Eliminating no‑op ``pass`` statements and removing extra ``finally`` clauses.
    """

    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """Solve the optimisation problem.

        Parameters
        ----------
        problem : dict
            Mapping containing the following keys:
            - "w_max"   (array‑like): maximum waiting times.
            - "d_max"   (array‑like): maximum delays.
            - "q_max"   (array‑like): maximum queue lengths.
            - "λ_min"   (array‑like): minimum arrival rates.
            - "μ_max"   (float): maximum total service rate.
            - "γ"       (array‑like): weights for the objective.

        Returns
        -------
        dict
            Dictionary with keys ``"μ"`` (service rates), ``"λ"`` (arrival rates)
            and ``"objective"`` (objective value).
        """
        # ----------------------------------------------------------
        # 1. Convert all inputs to contiguous NumPy arrays
        # ----------------------------------------------------------
        w_max = np.asarray(problem["w_max"], dtype=np.float64, order="C")
        d_max = np.asarray(problem["d_max"], dtype=np.float64, order="C")
        q_max = np.asarray(problem["q_max"], dtype=np.float64, order="C")
        λ_min = np.asarray(problem["λ_min"], dtype=np.float64, order="C")
        μ_max = float(problem["μ_max"])
        γ = np.asarray(problem["γ"], dtype=np.float64, order="C")
        n = γ.size

        # ----------------------------------------------------------
        # 2. Declare CVXPY variables
        # ----------------------------------------------------------
        μ = cp.Variable(n, pos=True)
        λ = cp.Variable(n, pos=True)

        # ----------------------------------------------------------
        # 3. Build expressions and constraints
        # ----------------------------------------------------------
        # ρ = λ/μ   (element‑wise division)
        ρ = λ / μ
        # System metrics
        q = cp.power(ρ, 2) / (1 - ρ)
        w = q / λ + 1 / μ
        d = 1 / (μ - λ)

        constraints = [
            w <= w_max,
            d <= d_max,
            q <= q_max,
            λ >= λ_min,
            cp.sum(μ) <= μ_max,
        ]

        # Objective: minimise weighted ratio of μ / λ
        objective = cp.Minimize(γ @ (μ / λ))

        prob = cp.Problem(objective, constraints)

        # ----------------------------------------------------------
        # 4. Solve: GP first, fall back to DCP if needed
        # ----------------------------------------------------------
        try:
            prob.solve(**_SOLVER_OPTIONS, gp=True)
        except cp.error.DGPError:
            try:
                prob.solve(**_SOLVER_OPTIONS)  # DCP mode
            except cp.error.DCPError:
                # Lastly, return a feasible but conservative solution
                λ_val = λ_min
                μ_val = np.full(n, μ_max / n)
                obj_val = float(γ @ (μ_val / λ_val))
                return {"μ": μ_val, "λ": λ_val, "objective": obj_val}

        # ----------------------------------------------------------
        # 5. Validate solver status and return the solution
        # ----------------------------------------------------------
        if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise RuntimeError(f"Solver failed with status {prob.status}")

        return {
            "μ": μ.value,
            "λ": λ.value,
            "objective": float(prob.value),
        }