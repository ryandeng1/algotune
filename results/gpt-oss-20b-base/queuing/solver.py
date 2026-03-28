import numpy as np
from scipy.optimize import minimize

__all__ = ['Solver']

class Solver:
    """Fast convex solver for the M/M/1 queue optimization problem."""
    def solve(self, problem: dict[str, any]) -> dict[str, any]:
        """
        Parameters
        ----------
        problem : dict
            Dictionary containing all problem data.
            Keys are the same as in the cvxpy version.

        Returns
        -------
        dict
            {'μ': μ_opt, 'λ': λ_opt, 'objective': obj_val}
        """
        # ---------- Data ---------- #
        w_max = np.asarray(problem['w_max'], dtype=float)
        d_max = np.asarray(problem['d_max'], dtype=float)
        q_max = np.asarray(problem['q_max'], dtype=float)
        λ_min = np.asarray(problem['λ_min'], dtype=float)
        μ_max = float(problem['μ_max'])
        γ   = np.asarray(problem['γ'], dtype=float)
        n    = γ.size

        # ---------- Helper functions ----------
        def objective(x):
            """μ/λ weighted sum."""
            μ = x[:n]
            λ = x[n:]
            return np.dot(γ, μ / λ)

        def weight_constraint(x):
            μ = x[:n]
            λ = x[n:]
            ρ   = λ / μ
            q   = ρ**2 / (1 - ρ)
            w   = q / λ + 1 / μ
            return w - w_max

        def delay_constraint(x):
            μ = x[:n]
            λ = x[n:]
            return 1 / (μ - λ) - d_max

        def queue_constraint(x):
            μ = x[:n]
            λ = x[n:]
            ρ   = λ / μ
            q   = ρ**2 / (1 - ρ)
            return q - q_max

        def lambda_lower(x):
            λ = x[n:]
            return λ - λ_min

        def mu_sum(x):
            μ = x[:n]
            return μ.sum() - μ_max

        # ---------- Initial guess ----------
        μ0 = np.full(n, μ_max / n)
        λ0 = np.copy(λ_min)
        # make sure λ < μ
        λ0 = np.minimum(λ0, μ0 - 1e-2)
        x0  = np.concatenate([μ0, λ0])

        # ---------- Constraints ----------
        cons = [
            {'type': 'ineq', 'fun': weight_constraint},
            {'type': 'ineq', 'fun': delay_constraint},
            {'type': 'ineq', 'fun': queue_constraint},
            {'type': 'ineq', 'fun': lambda_lower},
            {'type': 'ineq', 'fun': mu_sum},
        ]

        bounds = [(1e-8, None)] * (2 * n)  # μ, λ > 0

        # ---------- Solve ----------
        res = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'ftol': 1e-9, 'gtol': 1e-9, 'maxiter': 5000}
        )

        if not res.success:
            raise RuntimeError(f"Optimization failed: {res.message}")

        μ_opt = res.x[:n]
        λ_opt = res.x[n:]
        obj_val = objective(res.x)

        return {'μ': μ_opt, 'λ': λ_opt, 'objective': float(obj_val)}