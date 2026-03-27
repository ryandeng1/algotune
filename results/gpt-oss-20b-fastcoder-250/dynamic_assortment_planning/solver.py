from typing import Any, List, Dict
from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem: Dict[str, Any]) -> List[int]:
        """
        Solve the DAP exactly with a binary integer program (CP‑SAT).

        Returns a list with one optimal offer per period.  If the solver
        cannot find a feasible solution the list contains -1 for every
        period.
        """
        T: int = problem["T"]
        N: int = problem["N"]
        prices: List[float] = problem["prices"]
        capacities: List[int] = problem["capacities"]
        probs: List[List[float]] = problem["probs"]

        model = cp_model.CpModel()

        # Decision variables: x[t][i] = 1 ⇔ offer product i in period t
        x = [[model.NewBoolVar(f"x_{t}_{i}") for i in range(N)] for t in range(T)]

        # *At most one product per period*
        for t in range(T):
            model.Add(sum(x[t]) <= 1)

        # *Capacity limits for each product*
        for i in range(N):
            model.Add(sum(x[t][i] for t in range(T)) <= capacities[i])

        # *Objective: expected revenue*
        obj = sum(prices[i] * probs[t][i] * x[t][i]
                  for t in range(T)
                  for i in range(N))
        model.Maximize(obj)

        solver = cp_model.CpSolver()
        status = solver.Solve(model)

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return [-1] * T

        offer = [-1] * T
        for t in range(T):
            for i in range(N):
                if solver.Value(x[t][i]) == 1:
                    offer[t] = i
                    break

        return offer