from ortools.sat.python import cp_model
from typing import Any, List

class Solver:
    def solve(self, problem: dict[str, Any]) -> List[int]:
        """
        Solve the DAP exactly with a binary integer program (CP‑SAT).

        Returns
        -------
        List[int]
            offer[t] ∈ {‑1,0,…,N−1}.  ‑1 ⇒ offer nothing in period t.
        """
        T, N = problem["T"], problem["N"]
        prices = problem["prices"]
        capacities = problem["capacities"]
        probs = problem["probs"]

        model = cp_model.CpModel()
        x = {(t, i): model.NewBoolVar(f"x_{t}_{i}") for t in range(T) for i in range(N)}

        # At most one offer per time period
        for t in range(T):
            model.Add(sum(x[t, i] for i in range(N)) <= 1)

        # Capacity constraints for each offer
        for i in range(N):
            model.Add(sum(x[t, i] for t in range(T)) <= capacities[i])

        # Objective: maximise expected revenue
        model.Maximize(
            sum(prices[i] * probs[t][i] * x[t, i] for t in range(T) for i in range(N))
        )

        solver = cp_model.CpSolver()
        status = solver.Solve(model)

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return [-1] * T

        offer = []
        for t in range(T):
            chosen = -1
            for i in range(N):
                if solver.Value(x[t, i]):
                    chosen = i
                    break
            offer.append(chosen)
        return offer