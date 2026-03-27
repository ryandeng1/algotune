from ortools.sat.python import cp_model
from typing import List

class Solver:
    def solve(self, problem: List[List[int]]) -> List[int]:
        """Solve the TSP problem using CP‑SAT.

        Args:
            problem: Distance matrix.

        Returns:
            Tour starting and ending at city 0 or an empty list on failure.
        """
        n = len(problem)
        if n <= 1:
            return [0, 0]

        model = cp_model.CpModel()

        # Edge variables: one for each directed pair (i, j) with i != j
        x = {}
        for i in range(n):
            for j in range(n):
                if i != j:
                    x[(i, j)] = model.NewBoolVar(f"x[{i},{j}]")

        # Circuit constraint modelling a tour that visits all cities once
        model.AddCircuit([(u, v, x[(u, v)]) for (u, v) in x])

        # Objective: minimize total travel cost
        cost = sum(problem[i][j] * x[(i, j)]
                   for (i, j) in x)
        model.Minimize(cost)

        # Solver
        solver = cp_model.CpSolver()
        solver.parameters.log_search_progress = True
        status = solver.Solve(model)

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            tour = []
            current = 0
            for _ in range(n):
                tour.append(current)
                for nxt in range(n):
                    if current != nxt and solver.Value(x[(current, nxt)]) == 1:
                        current = nxt
                        break
            tour.append(0)
            return tour
        return []