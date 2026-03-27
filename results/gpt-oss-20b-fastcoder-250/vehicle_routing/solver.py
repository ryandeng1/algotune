from typing import Any, List
from ortools.sat.python import cp_model


class Solver:
    def solve(self, problem: dict[str, Any]) -> List[List[int]]:
        """
        Optimize VRP solution with CP-SAT.
        """
        D = problem["D"]
        K = problem["K"]
        depot = problem["depot"]
        n = len(D)

        model = cp_model.CpModel()

        # Create arc variables in a 2‑D list for fast access
        x = [[model.NewBoolVar(f"x_{i}_{j}") if i != j else None for j in range(n)]
             for i in range(n)]

        # Node entry/exit constraints (except depot)
        for i in range(n):
            if i == depot:
                continue
            model.Add(sum(x[j][i] for j in range(n) if j != i) == 1)
            model.Add(sum(x[i][j] for j in range(n) if j != i) == 1)

        # Depot has exactly K departures and K arrivals
        model.Add(sum(x[depot][j] for j in range(n) if j != depot) == K)
        model.Add(sum(x[i][depot] for i in range(n) if i != depot) == K)

        # MTZ subtour elimination variables
        u = [model.NewIntVar(1, n - 1, f"u_{i}") if i != depot else None for i in range(n)]
        for i in range(n):
            if i == depot:
                continue
            for j in range(n):
                if j == depot or i == j:
                    continue
                model.Add(u[i] + 1 <= u[j] + (n - 1) * (1 - x[i][j]))

        # Objective: minimize total distance
        model.Minimize(sum(D[i][j] * x[i][j]
                           for i in range(n) for j in range(n) if i != j))

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 60.0  # time limit
        status = solver.Solve(model)

        if status != cp_model.OPTIMAL and status != cp_model.FEASIBLE:
            return []

        # Reconstruct routes from the solution
        routes = []
        for j in range(n):
            if j == depot:
                continue
            if solver.Value(x[depot][j]):
                route = [depot, j]
                cur = j
                while cur != depot:
                    # next node in the route
                    nxt = next(k for k in range(n) if k != cur and solver.Value(x[cur][k]))
                    route.append(nxt)
                    cur = nxt
                routes.append(route)
        return routes