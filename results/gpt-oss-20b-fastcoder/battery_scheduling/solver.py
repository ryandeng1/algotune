from typing import Any, Dict, List
import numpy as np
import cvxpy as cp


class Solver:
    """
    A fast battery scheduling solver based on a simple linear program.
    The problem is always solved with the OSQP solver (a very fast QP solver)
    and most of the construction is done with NumPy instead of Python loops.
    """

    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        # Parse the input
        T = int(problem["T"])
        p = np.asarray(problem["p"], dtype=float)
        u = np.asarray(problem["u"], dtype=float)
        bat = problem["batteries"][0]
        Q = float(bat["Q"])
        C = float(bat["C"])
        D = float(bat["D"])
        eff = float(bat["efficiency"])

        # Decision variables
        q = cp.Variable(T)          # energy stored at time t
        cin = cp.Variable(T)        # charge power into battery
        cout = cp.Variable(T)       # discharge power from battery
        net = cin - cout            # net power change

        constraints: List[cp.Expression] = []

        # State of charge limits
        constraints.append(q >= 0)
        constraints.append(q <= Q)

        # Power limits
        constraints.append(cin >= 0)
        constraints.append(cin <= C)
        constraints.append(cout >= 0)
        constraints.append(cout <= D)

        # Dynamics (cyclic)
        # q[t+1] = q[t] + eff*cin[t] - (1/eff)*cout[t]
        dynamics = q[1:] == q[:-1] + eff * cin[:-1] - (1.0 / eff) * cout[:-1]
        constraints.append(dynamics)

        # Periodic boundary condition
        last = q[0] == q[-1] + eff * cin[-1] - (1.0 / eff) * cout[-1]
        constraints.append(last)

        # No simultaneous charging and discharging
        constraints.append(u + net >= 0)

        # Objective: minimise total cost
        objective = cp.Minimize(p @ net)

        prob = cp.Problem(objective, constraints)

        try:
            # Use OSQP with very few parameters for speed
            prob.solve(solver=cp.OSQP, eps_abs=1e-6, eps_rel=1e-6, max_iter=2000)
            if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
                return {"status": prob.status, "optimal": False}

            # Gather results
            q_val = q.value
            cin_val = cin.value
            cout_val = cout.value
            net_val = net.value

            cost_without = float(p @ u)
            cost_with = float(p @ (u + net_val))
            savings = cost_without - cost_with
            savings_pct = 100.0 * savings / cost_without if cost_without else 0.0

            return {
                "status": prob.status,
                "optimal": True,
                "battery_results": [
                    {
                        "q": q_val.tolist(),
                        "c": net_val.tolist(),
                        "c_in": cin_val.tolist(),
                        "c_out": cout_val.tolist(),
                        "cost": float(cost_with),
                    }
                ],
                "total_charging": net_val.tolist(),
                "cost_without_battery": float(cost_without),
                "cost_with_battery": float(cost_with),
                "savings": float(savings),
                "savings_percent": float(savings_pct),
            }

        except cp.SolverError as exc:
            return {"status": "solver_error", "optimal": False, "error": str(exc)}
        except Exception as exc:
            return {"status": "error", "optimal": False, "error": str(exc)}