from typing import Any, List, Dict
from ortools.sat.python import cp_model

def solve(problem: Dict[str, Any]) -> List[List[int]]:
    """Solve a simple vehicle routing problem with a fixed number of vehicles
    using OR-Tools CP‑SAT solver. The returned routes start and end at the
    depot."""
    D = problem["D"]
    K = problem["K"]
    depot = problem["depot"]
    n = len(D)

    # ----- Model -----
    model = cp_model.CpModel()

    # binary variables: 1 if the arc i->j is used
    x = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                x[i, j] = model.NewBoolVar(f"x_{i}_{j}")

    # each non-depot node has one incoming and one outgoing arc
    for i in range(n):
        if i == depot:
            continue
        model.Add(sum(x[j, i] for j in range(n) if j != i) == 1)
        model.Add(sum(x[i, j] for j in range(n) if j != i) == 1)

    # exactly K vehicles leave and return to the depot
    model.Add(sum(x[depot, j] for j in range(n) if j != depot) == K)
    model.Add(sum(x[i, depot] for i in range(n) if i != depot) == K)

    # Sub‑tour elimination (MTZ)
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
            model.Add(u[i] + 1 <= u[j] + (n - 1) * (1 - x[i, j]))

    # objective: minimize total distance
    model.Minimize(sum(D[i][j] * x[i, j] for i, j in x))

    # ----- Solve -----
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    # ----- Extract routes -----
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        routes = []
        for j in range(n):
            if j == depot or solver.Value(x[depot, j]) == 0:
                continue
            route = [depot, j]
            cur = j
            while cur != depot:
                for k in range(n):
                    if cur != k and solver.Value(x[cur, k]) == 1:
                        route.append(k)
                        cur = k
                        break
            routes.append(route)
        return routes
    return []