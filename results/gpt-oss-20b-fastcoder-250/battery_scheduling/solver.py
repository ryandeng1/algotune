import cvxpy as cp
import numpy as np
from typing import Any


class Solver:
    def solve(self, problem: dict) -> dict:
        T = int(problem["T"])
        p = np.asarray(problem["p"], dtype=np.float64)
        u = np.asarray(problem["u"], dtype=np.float64)

        battery = problem["batteries"][0]
        Q, C, D, eff = (
            float(battery["Q"]),
            float(battery["C"]),
            float(battery["D"]),
            float(battery["efficiency"]),
        )

        # Decision variables
        q    = cp.Variable(T, nonneg=True)          # Stored energy
        cin  = cp.Variable(T, nonneg=True)         # Charging rate
        cout = cp.Variable(T, nonneg=True)         # Discharging rate

        # Cap charging/discharging rates
        constraints = [
            cin   <= C,
            cout  <= D,
            q     <= Q,
        ]

        # Battery dynamics – use a companion matrix
        eff_in  = eff
        eff_out = 1/eff
        _A = np.eye(T, dtype=np.float64)
        _B = np.zeros((T, T), dtype=np.float64)
        for t in range(T - 1):
            _B[t + 1, t]   = 1
            _A[t + 1, t]   = -eff_in
            _A[t + 1, t]   = -eff_in  # will be eliminated by equality below
        # Actually easier to build dynamics vectorised
        # q[t+1] == q[t] + eff*cin[t] - (1/eff)*cout[t]
        dyn_eq = q[1:] - q[:-1] - eff * cin[:-1] + eff_out * cout[:-1]
        constraints.append(dyn_eq == 0)

        # Cyclic constraint: end with same state as start
        last_eq = q[0] - q[-1] - eff * cin[-1] + eff_out * cout[-1]
        constraints.append(last_eq == 0)

        # Hard constraint: no power sent back to the grid
        constraints.append(u + cin - cout >= 0)

        # Objective – cheaper to spend pre‑computations
        obj = cp.Minimize(p @ (cin - cout))

        prob = cp.Problem(obj, constraints)

        try:
            prob.solve(solver=cp.ECOS, verbose=False)
        except cp.SolverError as e:
            return {"status": "solver_error", "optimal": False, "error": str(e)}
        except Exception as e:
            return {"status": "error", "optimal": False, "error": str(e)}

        if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            return {"status": prob.status, "optimal": False}

        q_val   = q.value
        cin_val = cin.value
        cout_val= cout.value
        c_net   = cin_val - cout_val

        cost_without = float(p @ u)
        cost_with    = float(p @ (u + c_net))
        savings      = cost_without - cost_with

        return {
            "status": prob.status,
            "optimal": True,
            "battery_results": [
                {
                    "q": q_val.tolist(),
                    "c": c_net.tolist(),
                    "c_in": cin_val.tolist(),
                    "c_out": cout_val.tolist(),
                    "cost": cost_with,
                }
            ],
            "total_charging": c_net.tolist(),
            "cost_without_battery": cost_without,
            "cost_with_battery": cost_with,
            "savings": savings,
            "savings_percent": 100 * savings / cost_without if cost_without else 0.0,
        }