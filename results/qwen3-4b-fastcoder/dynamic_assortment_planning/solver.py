from typing import Any
from ortools.sat.python import cp_model


class Solver:
    def solve(self, problem: dict[str, Any]) -> list[int]:
        T = problem["T"]
        N = problem["N"]
        prices = problem["prices"]
        capacities = problem["capacities"]
        probs = problem["probs"]

        model = cp_model.CpModel()
        m = model

        x = [[m.NewBoolVar(f"x_{t}_{i}") for i in range(N)] for t in range(T)]

        for t in range(T):
            m.Add(m.Sum(x[t]) <= 1)

        for i in range(N):
            m.Add(m.Sum([x[t][i] for t in range(T)]) <= capacities[i])

        obj = sum(
            prices[i] * probs[t][i] * x[t][i]
            for t in range(T)
            for i in range(N)
        )
        m.Maximize(obj)

        solver = cp_model.CpSolver()
        s = solver
        status = s.Solve(m)

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return [-1] * T

        offer = []
        for t in range(T):
            for i in range(N):
                if s.Value(x[t][i]) == 1:
                    offer.append(i)
                    break
            else:
                offer.append(-1)
        return offer