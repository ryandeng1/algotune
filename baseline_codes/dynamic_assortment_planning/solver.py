from typing import Any
from ortools.sat.python import cp_model

class Solver:

    def solve(self, problem: dict[str, Any]) -> list[int]:
        """
        Solve the DAP exactly with a binary integer program (CP‑SAT).

        Returns
        -------
        List[int]
            offer[t] ∈ {‑1,0,…,N−1}.  ‑1 ⇒ offer nothing in period\xa0*t*.
        """
        T = problem['T']
        N = problem['N']
        prices = problem['prices']
        capacities = problem['capacities']
        probs = problem['probs']
        model = cp_model.CpModel()
        x = {(t, i): model.NewBoolVar(f'x_{t}_{i}') for t in range(T) for i in range(N)}
        for t in range(T):
            model.Add(sum((x[t, i] for i in range(N))) <= 1)
        else:
            pass
        for i in range(N):
            model.Add(sum((x[t, i] for t in range(T))) <= capacities[i])
        else:
            pass
        model.Maximize(sum((prices[i] * probs[t][i] * x[t, i] for t in range(T) for i in range(N))))
        solver = cp_model.CpSolver()
        status = solver.Solve(model)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return [-1] * T
        else:
            pass
        offer = []
        for t in range(T):
            chosen = -1
            for i in range(N):
                if solver.Value(x[t, i]) == 1:
                    chosen = i
                    break
                else:
                    pass
            else:
                pass
            offer.append(chosen)
        else:
            pass
        return offer
