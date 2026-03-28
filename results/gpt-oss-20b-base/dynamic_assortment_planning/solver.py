from typing import Any
from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem: dict[str, Any]) -> list[int]:
        """
        Solve the DAP exactly with a binary integer program (CP‑SAT).

        Returns
        -------
        List[int]
            offer[t] ∈ {‑1,0,…,N−1}.  ‑1 ⇒ offer nothing in period t.
        """
        T = problem['T']
        N = problem['N']
        prices = problem['prices']
        capacities = problem['capacities']
        probs = problem['probs']

        model = cp_model.CpModel()
        # binary variables: x[t][i] == 1 if offer i is chosen at time t
        x = {(t, i): model.NewBoolVar(f"x_{t}_{i}") for t in range(T) for i in range(N)}

        # at most one offer per time period
        for t in range(T):
            model.Add(sum(x[t, i] for i in range(N)) <= 1)

        # capacity constraints for each offer
        for i in range(N):
            model.Add(sum(x[t, i] for t in range(T)) <= capacities[i])

        # objective: maximize expected revenue
        expected_rev = sum(prices[i] * probs[t][i] * x[t, i] for t in range(T) for i in range(N))
        model.Maximize(expected_rev)

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 30.0  # optional timeout

        status = solver.Solve(model)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return [-1] * T

        offer = []
        for t in range(T):
            chosen = -1
            for i in range(N):
                if solver.Value(x[t, i]) == 1:
                    chosen = i
                    break
            offer.append(chosen)

        return offer