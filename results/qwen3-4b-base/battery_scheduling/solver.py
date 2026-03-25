import cvxpy as cp
import numpy as np

class Solver:
    def solve(self, problem: dict) -> dict:
        try:
            T = int(problem["T"])
            p = np.array(problem["p"])
            u = np.array(problem["u"])
            battery = problem["batteries"][0]
            Q = float(battery["Q"])
            C = float(battery["C"])
            D = float(battery["D"])
            efficiency = float(battery["efficiency"])
            
            q = cp.Variable(T)
            c_in = cp.Variable(T)
            c_out = cp.Variable(T)
            c = c_in - c_out
            
            constraints = [
                q >= 0,
                q <= Q,
                c_in >= 0,
                c_out >= 0,
                c_in <= C,
                c_out <= D,
                q[1:] == q[:-1] + efficiency * c_in[:-1] - (1 / efficiency) * c_out[:-1],
                q[0] == q[T-1] + efficiency * c_in[T-1] - (1 / efficiency) * c_out[T-1],
                u + c >= 0
            ]
            
            objective = cp.Minimize(p @ c)
            prob = cp.Problem(objective, constraints)
            prob.solve(solver=cp.SCS, verbose=False)
            
            if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                return {"status": prob.status, "optimal": False}
            
            c_net = c_in.value - c_out.value
            cost_without_battery = float(p @ u)
            cost_with_battery = float(p @ (u + c_net))
            savings = cost_without_battery - cost_with_battery
            
            return {
                "status": prob.status,
                "optimal": True,
                "battery_results": [
                    {
                        "q": q.value.tolist(),
                        "c": c_net.tolist(),
                        "c_in": c_in.value.tolist(),
                        "c_out": c_out.value.tolist(),
                        "cost": cost_with_battery,
                    }
                ],
                "total_charging": c_net.tolist(),
                "cost_without_battery": cost_without_battery,
                "cost_with_battery": cost_with_battery,
                "savings": savings,
                "savings_percent": float(100 * savings / cost_without_battery),
            }
        except cp.SolverError as e:
            return {"status": "solver_error", "optimal": False, "error": str(e)}
        except Exception as e:
            return {"status": "error", "optimal": False, "error": str(e)}
