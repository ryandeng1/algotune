from typing import Any, List
from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem: dict[str, Any]) -> List[List[int]]:
        """
        Solve a Capacitated Vehicle Routing Problem (CVRP) with a single depot
        assuming unlimited capacity.  The objective is to minimise the total
        travel cost.  The returned routes are lists of node indices starting
        and ending at the depot.
        """
        D = problem["D"]
        K = problem["K"]
        depot = problem["depot"]
        n = len(D)

        # ---------- Build the model ----------
        model = cp_model.CpModel()

        # Edge variables
        x = {
            (i, j): model.NewBoolVar(f"x_{i}_{j}")
            for i in range(n)
            for j in range(n)
            if i != j
        }

        # Each non‑depot vertex has one predecessor and one successor
        for i in range(n):
            if i == depot:
                continue
            model.Add(sum(x[j, i] for j in range(n) if j != i) == 1)
            model.Add(sum(x[i, j] for j in range(n) if j != i) == 1)

        # Exactly K arcs leave/enter the depot
        model.Add(sum(x[depot, j] for j in range(n) if j != depot) == K)
        model.Add(sum(x[i, depot] for i in range(n) if i != depot) == K)

        # Sub‑tour elimination (Miller‑Tucker‑Zemlin)
        u = {
            i: model.NewIntVar(1, n - 1, f"u_{i}")
            for i in range(n)
            if i != depot
        }
        for i in range(n):
            if i == depot:
                continue
            for j in range(n):
                if j == depot or i == j:
                    continue
                model.Add(u[i] + 1 <= u[j] + (n - 1) * (1 - x[i, j]))

        # Objective: minimise total distance
        model.Minimize(
            sum(D[i][j] * x[i, j] for i in range(n) for j in range(n) if i != j)
        )

        # ---------- Solve ----------
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 60.0  # optional timeout
        status = solver.Solve(model)

        if status != cp_model.OPTIMAL:
            return []

        # ---------- Extract routes ----------
        routes: List[List[int]] = []
        for j in range(n):
            if j == depot or solver.Value(x[depot, j]) == 0:
                continue
            route = [depot, j]
            cur = j
            while cur != depot:
                # find the outgoing edge from cur
                for k in range(n):
                    if cur != k and solver.Value(x[cur, k]) == 1:
                        route.append(k)
                        cur = k
                        break
            routes.append(route)

        return routes