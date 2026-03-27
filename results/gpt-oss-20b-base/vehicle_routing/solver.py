from typing import Any
from ortools.sat.python import cp_model


class Solver:
    def solve(self, problem: dict[str, Any]) -> list[list[int]]:
        D = problem["D"]
        K = problem["K"]
        depot = problem["depot"]
        n = len(D)

        model = cp_model.CpModel()

        # Variables: x[i][j] == 1 if arc i->j is used
        x = [[None] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j:
                    x[i][j] = model.NewBoolVar(f"x_{i}_{j}")

        # Flow (degree) constraints for every node
        for i in range(n):
            if i == depot:
                continue
            model.Add(sum(x[j][i] for j in range(n) if j != i) == 1)
            model.Add(sum(x[i][j] for j in range(n) if j != i) == 1)

        # Depot has exactly K departures and K arrivals
        model.Add(sum(x[depot][j] for j in range(n) if j != depot) == K)
        model.Add(sum(x[i][depot] for i in range(n) if i != depot) == K)

        # MTZ subtour elimination
        u = [None] * n
        for i in range(n):
            if i == depot:
                continue
            u[i] = model.NewIntVar(1, n - 1, f"u_{i}")
        for i in range(n):
            if i == depot:
                continue
            for j in range(n):
                if j == depot or i == j:
                    continue
                # u_i + 1 <= u_j + (n-1)*(1-x_ij)
                model.Add(u[i] + 1 <= u[j] + (n - 1) * (1 - x[i][j]))

        # Objective: minimize total distance
        objective = sum(D[i][j] * x[i][j] for i in range(n) for j in range(n) if i != j)
        model.Minimize(objective)

        solver = cp_model.CpSolver()
        status = solver.Solve(model)

        if status == cp_model.OPTIMAL:
            routes = []
            for start in range(n):
                if start != depot and solver.Value(x[depot][start]):
                    route = [depot, start]
                    cur = start
                    while cur != depot:
                        for nxt in range(n):
                            if cur != nxt and solver.Value(x[cur][nxt]):
                                route.append(nxt)
                                cur = nxt
                                break
                    routes.append(route)
            return routes
        return []