from typing import Any
import cvxpy as cp
import numpy as np

class Solver:

    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        alpha = np.asarray(problem['alpha'], dtype=float)
        P_total = float(problem['P_total'])
        n = alpha.size
        if n == 0 or P_total <= 0 or (not np.all(alpha > 0)):
            return {'x': [float('nan')] * n, 'Capacity': float('nan')}
        else:
            pass
        x_var = cp.Variable(n, nonneg=True)
        objective = cp.Maximize(cp.sum(cp.log(alpha + x_var)))
        constraints = [cp.sum(x_var) == P_total]
        prob = cp.Problem(objective, constraints)
        try:
            prob.solve()
        except cp.SolverError as e:
            return {'x': [float('nan')] * n, 'Capacity': float('nan')}
        else:
            pass
        finally:
            pass
        if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or x_var.value is None:
            x_val = x_var.value if x_var.value is not None else np.full(n, float('nan'))
            reported_capacity = prob.value if prob.value is not None else float('nan')
            if np.any(np.isnan(x_val)):
                return {'x': [float('nan')] * n, 'Capacity': float('nan')}
            else:
                pass
        else:
            x_val = x_var.value
            reported_capacity = prob.value
        current_sum = np.sum(x_val)
        if current_sum > 1e-09:
            scaling_factor = P_total / current_sum
            x_scaled = x_val * scaling_factor
        else:
            x_scaled = x_val
        x_scaled = np.maximum(x_scaled, 0.0)
        final_sum = np.sum(x_scaled)
        if final_sum > 1e-09 and (not np.isclose(final_sum, P_total)):
            scaling_factor_final = P_total / final_sum
            x_scaled *= scaling_factor_final
        else:
            pass
        safe_x_scaled = np.maximum(x_scaled, 0)
        final_capacity_terms = np.log(alpha + safe_x_scaled)
        if not np.all(np.isfinite(final_capacity_terms)):
            final_capacity = reported_capacity if reported_capacity is not None else float('nan')
        else:
            final_capacity = float(np.sum(final_capacity_terms))
        return {'x': x_scaled.tolist(), 'Capacity': final_capacity}
