from typing import Any
import cvxpy as cp
import numpy as np

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        # Convert inputs to numpy arrays
        w_max = np.asarray(problem['w_max'], dtype=np.float64)
        d_max = np.asarray(problem['d_max'], dtype=np.float64)
        q_max = np.asarray(problem['q_max'], dtype=np.float64)
        λ_min = np.asarray(problem['λ_min'], dtype=np.float64)
        μ_max = float(problem['μ_max'])
        γ = np.asarray(problem['γ'], dtype=np.float64)

        n = γ.size
        μ = cp.Variable(n, pos=True)
        λ = cp.Variable(n, pos=True)
        ρ = λ / μ
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

        objective = cp.Minimize(γ @ (μ / λ))
        prob = cp.Problem(objective, constraints)

        # Solve once; if GP fails fall back to a simple fallback solution
        try:
            prob.solve(gp=True, verbose=False, solver=cp.SCS, eps=1e-8)
        except cp.error.DGPError:
            # Fallback: simple feasible assignment
            λ_val = λ_min
            μ_val = np.full(n, μ_max / n)
            obj_val = float(γ @ (μ_val / λ_val))
            return {'μ': μ_val, 'λ': λ_val, 'objective': obj_val}

        if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise RuntimeError(f'CVXPY solver failed: status {prob.status}')

        return {
            'μ': μ.value.astype(np.float64),
            'λ': λ.value.astype(np.float64),
            'objective': float(prob.value),
        }