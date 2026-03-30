# solver.py
from typing import Any, List
from ortools.sat.python import cp_model


class Solver:
    """Fast exact solver for the Dynamic Allocation Problem using OR‑Tools CP‑SAT."""

    def solve(self, problem: dict[str, Any]) -> List[int]:
        """
        Solve the DAP exactly with a binary integer program (CP‑SAT).

        Parameters
        ----------
        problem : dict
            Must contain ``T`` (int), ``N`` (int), ``prices`` (list[float]),
            ``capacities`` (list[int]) and ``probs`` (list[list[float]]).
            ``probs[t][i]`` is the probability that offer *i* appears at time *t*.

        Returns
        -------
        List[int]
            ``offer[t]`` ∈ {‑1, …, N‑1}.  ‑1 bedeutet, dass in Periode *t* nichts
            angeboten wird.
        """
        T = problem["T"]
        N = problem["N"]
        prices = problem["prices"]
        capacities = problem["capacities"]
        probs = problem["probs"]

        model = cp_model.CpModel()

        # Decision variables: x[t,i] == 1 iff offer i is selected in period t
        x = [
            [model.NewBoolVar(f"x_{t}_{i}") for i in range(N)] for t in range(T)
        ]

        # At most one offer per period
        for t in range(T):
            model.Add(sum(x[t]) <= 1)

        # Respect capacities of each offer
        for i in range(N):
            model.Add(sum(x[t][i] for t in range(T)) <= capacities[i])

        # Objective: maximise expected revenue
        expected_rev = sum(
            prices[i] * probs[t][i] * x[t][i] for t in range(T) for i in range(N)
        )
        model.Maximize(expected_rev)

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 10.0  # avoid long‑running exhaustive search
        status = solver.Solve(model)

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            # No feasible solution found – return all -1
            return [-1] * T

        # Build the solution list
        offer: List[int] = []
        for t in range(T):
            chosen = -1
            # Since at most one variable is 1 per period, we can break early
            for i in range(N):
                if solver.Value(x[t][i]):
                    chosen = i
                    break
            offer.append(chosen)

        return offer