from typing import Any
import cvxpy as cp
import numpy as np

class Solver:
    def solve(self, problem: dict) -> dict:
        T = int(problem['T'])
        p = np.array(problem['p'])
        u = np.array(problem['u'])
        battery = problem['batteries'][0]
        Q, C, D = float(battery['Q']), float(battery['C']), float(battery['D'])
        eff = float(battery['efficiency'])

        q = cp.Variable(T)
        c_in = cp.Variable(T, nonneg=True)
        c_out = cp.Variable(T, nonneg=True)

        constraints = [
            q >= 0, q <= Q,
            c_in <= C, c_out <= D,
            u + (c_in - c_out) >= 0
        ]

        # State updates
        for t in range(T - 1):
            constraints.append(q[t + 1] == q[t] + eff * c_in[t] - c_in[t] / eff)
        constraints.append(q[0] == q[T - 1] + eff * c_in[T - 1] - c_in[T - 1] / eff)

        objective = cp.Minimize(p @ (c_in - c_out))
        prob = cp.Problem(objective, constraints)

        try:
            prob.solve(solver=cp.ECOS, verbose=False)
            status = prob.status
            if status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
                return {'status': status, 'optimal': False}

            c_net = (c_in.value - c_out.value).tolist()
            cost_without = float(p @ u)
            cost_with = float(p @ (u + c_net))
            savings = cost_without - cost_with
            return {
                'status': status,
                'optimal': True,
                'battery_results': [{
                    'q': q.value.tolist(),
                    'c': c_net,
                    'c_in': c_in.value.tolist(),
                    'c_out': c_out.value.tolist(),
                    'cost': cost_with
                }],
                'total_charging': c_net,
                'cost_without_battery': cost_without,
                'cost_with_battery': cost_with,
                'savings': savings,
                'savings_percent': 100 * savings / cost_without if cost_without else 0.0
            }
        except cp.SolverError as e:
            return {'status': 'solver_error', 'optimal': False, 'error': str(e)}
        except Exception as e:
            return {'status': 'error', 'optimal': False, 'error': str(e)}