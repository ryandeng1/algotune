# solver.py
import numpy as np
import pulp


class Solver:
    def solve(self, problem: dict) -> dict:
        # extract parameters
        T = int(problem["T"])
        p = np.array(problem["p"], dtype=float)
        u = np.array(problem["u"], dtype=float)
        bat = problem["batteries"][0]  # single battery
        Q, C, D, eff = (
            float(bat["Q"]),
            float(bat["C"]),
            float(bat["D"]),
            float(bat["efficiency"]),
        )

        # decision variables
        q = pulp.LpVariable.dicts("q", range(T), lowBound=0, upBound=Q)
        cin = pulp.LpVariable.dicts("cin", range(T), lowBound=0, upBound=C)
        cout = pulp.LpVariable.dicts("cout", range(T), lowBound=0, upBound=D)

        # problem definition
        prob = pulp.LpProblem("BatteryScheduling", pulp.LpMinimize)

        # objective: sum p_t * (cin_t - cout_t)
        prob += pulp.lpSum(p[t] * (cin[t] - cout[t]) for t in range(T))

        # battery dynamics
        for t in range(T - 1):
            prob += (
                q[t + 1]
                == q[t] + eff * cin[t] - (1.0 / eff) * cout[t]
            )
        # cyclic constraint
        prob += (
            q[0]
            == q[T - 1] + eff * cin[T - 1] - (1.0 / eff) * cout[T - 1]
        )

        # no power back to grid
        for t in range(T):
            prob += u[t] + (cin[t] - cout[t]) >= 0

        # solve
        prob.solve(pulp.PULP_CBC_CMD(msg=False))

        status = pulp.LpStatus[prob.status]
        if status != "Optimal":
            return {"status": status, "optimal": False}

        # extract results
        q_val = [float(q[t].value()) for t in range(T)]
        cin_val = [float(cin[t].value()) for t in range(T)]
        cout_val = [float(cout[t].value()) for t in range(T)]
        c_val = [cin_val[t] - cout_val[t] for t in range(T)]

        cost_without = float(p.dot(u))
        cost_with = float(p.dot(np.array(u) + np.array(c_val)))
        savings = cost_without - cost_with

        return {
            "status": status,
            "optimal": True,
            "battery_results": [
                {
                    "q": q_val,
                    "c": c_val,
                    "c_in": cin_val,
                    "c_out": cout_val,
                    "cost": cost_with,
                }
            ],
            "total_charging": c_val,
            "cost_without_battery": cost_without,
            "cost_with_battery": cost_with,
            "savings": savings,
            "savings_percent": 100.0 * savings / cost_without,
        }
