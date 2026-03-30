# solver.py
from typing import Any, List
from ortools.sat.python import cp_model


class Solver:
    """
    Vehicle routing solver using OR‑Tools CP‑SAT.

    This implementation keeps the model construction
    lightweight and uses Python generators to avoid
    building large intermediate data structures.
    """

    def solve(self, problem: dict[str, Any]) -> List[List[int]]:
        """Return K routes for the given VRP instance."""
        D = problem["D"]
        K = problem["K"]
        depot = problem["depot"]
        n = len(D)

        model = cp_model.CpModel()

        # Edge decision variables (i → j)
        x = {(i, j): model.NewBoolVar(f"x_{i}_{j}") for i in range(n) for j in range(n) if i != j}

        # Flow conservation (each node has one incoming and one outgoing edge)
        for i in range(n):
            if i == depot:
                continue
            model.Add(sum(x[(j, i)] for j in range(n) if j != i) == 1)
            model.Add(sum(x[(i, j)] for j in range(n) if j != i) == 1)

        # Exactly K vehicles depart/return to the depot
        model.Add(sum(x[(depot, j)] for j in range(n) if j != depot) == K)
        model.Add(sum(x[(i, depot)] for i in range(n) if i != depot) == K)

        # Sub‑tour elimination (Miller–Tucker–Zemlin)
        u = {i: model.NewIntVar(1, n - 1, f"u_{i}") for i in range(n) if i != depot}
        for i in range(n):
            if i == depot:
                continue
            for j in range(n):
                if j == depot or i == j:
                    continue
                model.Add(u[i] + 1 <= u[j] + (n - 1) * (1 - x[(i, j)]))

        # Objective: minimise total distance
        model.Minimize(sum(D[i][j] * x[(i, j)] for i, j in x))

        # Solve
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 30.0  # Prevent very long solves
        solver.parameters.num_search_workers = 8
        status = solver.Solve(model)

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return []

        # Extract routes
        routes: List[List[int]] = []
        visited_start = [False] * n
        for j in range(n):
            if j != depot and solver.Value(x[(depot, j)]) == 1 and not visited_start[j]:
                route = [depot, j]
                current = j
                visited_start[j] = True
                while current != depot:
                    next_node = next(
                        k
                        for k in range(n)
                        if k != current and solver.Value(x[(current, k)]) == 1
                    )
                    route.append(next_node)
                    current = next_node
                routes.append(route)

        return routes