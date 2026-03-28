from typing import Any, List
from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem: dict[str, Any]) -> List[int]:
        """
        Solve the DAP exactly with a binary integer program (CP‑SAT).

        Returns
        -------
        List[int]
            offer[t] ∈ {‑1,0,…,N−1}.  ‑1 ⇒ offer nothing in period t.
        """
        T, N = problem["T"], problem["N"]
        prices, capacities, probs = problem["prices"], problem["capacities"], problem["probs"]

        model = cp_model.CpModel()

        # Decision variables: x[t,i] = 1 if plant i is chosen in period t
        x = {(t, i): model.NewBoolVar(f"x_{t}_{i}") for t in range(T) for i in range(N)}

        # at most one offer per period
        for t in range(T):
            model.Add(sum(x[t, i] for i in range(N)) <= 1)

        # capacity constraints
        for i in range(N):
            model.Add(sum(x[t, i] for t in range(T)) <= capacities[i])

        # objective: maximize expected profit
        model.Maximize(
            sum(prices[i] * probs[t][i] * x[t, i] for t in range(T) for i in range(N))
        )

        solver = cp_model.CpSolver()
        status = solver.Solve(model)

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return [-1] * T

        # build solution
        offer = []
        for t in range(T):
            chosen = -1
            for i in range(N):
                if solver.Value(x[t, i]):
                    chosen = i
                    break
            offer.append(chosen)

        return offer