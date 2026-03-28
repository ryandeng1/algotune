import numpy as np
from typing import Any, Dict, List

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[float]] | None:
        """Solve the mean‑variance portfolio optimisation
        max μᵀx – γ xᵀΣx  s.t.  Σᵢxᵢ = 1 ,  x ≥ 0
        using an active‑set quadratic solver.
        """
        μ = np.asarray(problem["μ"], dtype=np.float64)
        Σ = np.asarray(problem["Σ"], dtype=np.float64)
        γ = float(problem["γ"])

        n = μ.size
        if Σ.shape != (n, n):
            return None

        # Pre‑compute inverses needed for the active set iterations
        try:
            Σ_inv = np.linalg.inv(Σ)
        except np.linalg.LinAlgError:
            return None

        # Pre‑compute vectors that will be reused
        Σ_inv_μ = Σ_inv @ μ
        Σ_inv_1 = Σ_inv @ np.ones(n)

        # Start with all indices in the active set
        active = np.arange(n, dtype=int)

        while True:
            k = active.size
            if k == 0:
                return None  # infeasible

            # Solve the unconstrained (equal‑constraint only) problem on active set
            # γ Σ_active w_active = 0.5 μ_active – λ 1_active
            # => w_active = (1/(2γ)) Σ_inv_active μ_active – λ Σ_inv_active 1_active
            # Find λ such that sum(w_active) = 1
            inv = Σ_inv[np.ix_(active, active)]
            μ_a = μ[active]
            ones_a = np.ones(k)

            A = inv @ ones_a
            b = inv @ μ_a

            # λ = (b - (1/(2γ))*onesᵀ A) / (onesᵀ A)
            denom = ones_a @ A
            if denom == 0:
                return None
            λ = (b @ ones_a - (1/(2*γ))) / denom

            w_a = (1/(2*γ)) * b - λ * A

            # Construct full w vector
            w_full = np.zeros(n, dtype=np.float64)
            w_full[active] = w_a

            # Check feasibility
            neg_mask = w_a < 0
            if not neg_mask.any():
                # Feasible; projection onto simplex already satisfied, so return
                return {"w": w_full.tolist()}

            # Remove the most negative variable from the active set and retry
            to_remove = active[neg_mask][np.argmin(w_a[neg_mask])]
            active = active[active != to_remove]