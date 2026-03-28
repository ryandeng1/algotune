from ortools.sat.python import cp_model
from typing import Any, List

class Solver:
    def solve(self, problem: dict[str, Any]) -> List[List[int]]:
        """
        Solve a Capacitated VRP (no capacity constraints) using OR-Tools CP-SAT.
        The solver is very compact and avoids Python overhead in model construction
        and solution parsing.
        """
        D = problem["D"]
        K = problem["K"]
        depot = problem["depot"]
        n = len(D)

        model = cp_model.CpModel()

        # Decision variables: x[i][j] = 1 if vehicle travels i->j
        x = [[None] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j:
                    x[i][j] = model.NewBoolVar(f"x_{i}_{j}")

        # Each non-depot node has one incoming and one outgoing edge
        for i in range(n):
            if i == depot:
                continue
            model.Add(sum(x[j][i] for j in range(n) if j != i) == 1)
            model.Add(sum(x[i][j] for j in range(n) if j != i) == 1)

        # Depot has exactly K outgoing and K incoming edges
        model.Add(sum(x[depot][j] for j in range(n) if j != depot) == K)
        model.Add(sum(x[i][depot] for i in range(n) if i != depot) == K)

        # Subtour elimination (Miller‑Tucker‑Zemlin)
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
                # u[i] + 1 <= u[j] + (n-1)*(1-x[i][j])
                model.Add(u[i] + 1 <= u[j] + (n - 1) * (1 - x[i][j]))

        # Objective: minimize total distance
        model.Minimize(sum(D[i][j] * x[i][j] for i in range(n) for j in range(n) if i != j))

        solver = cp_model.CpSolver()
        status = solver.Solve(model)

        if status != cp_model.OPTIMAL:
            return []

        # Extract routes
        routes = []
        for start in range(n):
            if start == depot:
                continue
            if solver.Value(x[depot][start]) == 1:
                route = [depot, start]
                current = start
                while current != depot:
                    for nxt in range(n):
                        if current != nxt and solver.Value(x[current][nxt]) == 1:
                            route.append(nxt)
                            current = nxt
                            break
                routes.append(route)

        return routes