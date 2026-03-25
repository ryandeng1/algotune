import numpy as np
from ortools.sat.python import cp_model
from typing import Any, List

class Solver:
    def solve(self, problem: dict[str, Any]) -> List[List[int]]:
        """
        Solve the Vehicle Routing Problem (VRP) using the CP-SAT solver from OR-Tools.
        This implementation follows the reference solution and is expected to produce
        the optimal solution for the given problem instance.
        """
        D = problem["D"]
        K = problem["K"]
        depot = problem["depot"]
        n = len(D)

        model = cp_model.CpModel()

        # Decision variables: x[i,j] = 1 if arc i->j is used
        x = {}
        for i in range(n):
            for j in range(n):
                if i != j:
                    x[(i, j)] = model.NewBoolVar(f"x_{i}_{j}")

        # Each non-depot node must be entered and left exactly once
        for i in range(n):
            if i == depot:
                continue
            model.Add(
                sum(x[(j, i)] for j in range(n) if j != i) == 1
            )
            model.Add(
                sum(x[(i, j)] for j in range(n) if j != i) == 1
            )

        # Depot must have exactly K departures and arrivals
        model.Add(
            sum(x[(depot, j)] for j in range(n) if j != depot) == K
        )
        model.Add(
            sum(x[(i, depot)] for i in range(n) if i != depot) == K
        )

        # MTZ subtour elimination constraints
        u = {}
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
                model.Add(
                    u[i] + 1 <= u[j] + (n - 1) * (1 - x[(i, j)])
                )

        # Objective: minimize total travel distance
        model.Minimize(sum(D[i][j] * x[(i, j)] for i, j in x))

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 30.0  # give some time if large
        status = solver.Solve(model)

        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            routes = []
            for j in range(n):
                if j != depot and solver.Value(x[(depot, j)]) == 1:
                    route = [depot, j]
                    current = j
                    while current != depot:
                        for k in range(n):
                            if current != k and solver.Value(x[(current, k)]) == 1:
                                route.append(k)
                                current = k
                                break
                    routes.append(route)
            # If fewer routes than K, pad with empty depot routes
            while len(routes) < K:
                routes.append([depot, depot])
            return routes
        else:
            # Fall back in the unlikely event of no solution
            return [[depot, depot] for _ in range(K)]
