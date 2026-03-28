from typing import List
from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem: List[List[int]]) -> List[int]:
        """
        Solve the TSP problem using OR‑Tools CP‑SAT.

        :param problem: Distance matrix as a list of lists.
        :return: A list representing the optimal tour, starting and ending at city 0.
        """
        n = len(problem)
        if n <= 1:
            return [0, 0]

        model = cp_model.CpModel()

        # Decision variables: x[i][j] == 1 iff we travel from i to j
        x = {}
        for i in range(n):
            for j in range(n):
                if i != j:
                    x[(i, j)] = model.NewBoolVar(f"x[{i},{j}]")

        # Circuit constraint: exactly one outgoing and one incoming edge per city
        path_edges = [(u, v, x[(u, v)]) for (u, v) in x]
        model.AddCircuit(path_edges)

        # Objective: minimize total travel cost
        obj = sum(problem[i][j] * x[(i, j)] for (i, j) in x)
        model.Minimize(obj)

        # Solver configuration
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 30.0   # give a generous time limit
        solver.parameters.num_search_workers = 2     # use 2 cores

        status = solver.Solve(model)

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            tour = [0]
            cur = 0
            while len(tour) < n:
                for nxt in range(n):
                    if cur != nxt and solver.Value(x[(cur, nxt)]) == 1:
                        tour.append(nxt)
                        cur = nxt
                        break
            tour.append(0)
            return tour
        return []