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

        x = [[model.NewBoolVar() for _ in range(N)] for _ in range(T)]

        for t in range(T):
            model.Add(sum(x[t]) <= 1)

        for i in range(N):
            model.Add(sum(x[t][i] for t in range(T)) <= capacities[i])

        model.Maximize(
            sum(prices[i] * probs[t][i] * x[t][i] for t in range(T) for i in range(N))
        )

        solver = cp_model.CpSolver()
        status = solver.Solve(model)

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return [-1] * T

        offer = []
        for t in range(T):
            chosen = -1
            for i in range(N):
                if solver.Value(x[t][i]) == 1:
                    chosen = i
                    break
            offer.append(chosen)
        return offer