from typing import Any, List
from ortools.sat.python import cp_model

def solve(problem: dict[str, Any]) -> List[List[int]]:
    """Solve a Capacitated Vehicle Routing Problem (CVRP) with no capacity limits
    using the CP‑SAT solver.  The solution is a list of K routes, each list of
    nodes starting and ending at the depot.
    """
    D = problem["D"]
    K = problem["K"]
    depot = problem["depot"]
    n = len(D)

    model = cp_model.CpModel()

    # Edge variables: 1 if arc (i, j) is used
    x = {
        (i, j): model.NewBoolVar(f"x_{i}_{j}")
        for i in range(n)
        for j in range(n)
        if i != j
    }

    # Degree constraints (each non‑depot node has exactly one incoming and one outgoing arc)
    for v in range(n):
        if v == depot:
            continue
        model.Add(sum(x[(i, v)] for i in range(n) if i != v) == 1)
        model.Add(sum(x[(v, j)] for j in range(n) if j != v) == 1)

    # Depot constraints: the number of vehicles (=K)
    model.Add(sum(x[(depot, j)] for j in range(n) if j != depot) == K)
    model.Add(sum(x[(i, depot)] for i in range(n) if i != depot) == K)

    # Subtour elimination (Miller‑Tucker‑Zemlin)
    u = {i: model.NewIntVar(1, n - 1, f"u_{i}") for i in range(n) if i != depot}
    for i in range(n):
        if i == depot:
            continue
        for j in range(n):
            if j == depot or j == i:
                continue
            model.Add(u[i] + 1 <= u[j] + (n - 1) * (1 - x[(i, j)]))

    # Objective: minimize total distance
    model.Minimize(sum(D[i][j] * x[(i, j)] for (i, j) in x))

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    # Extract routes
    if status == cp_model.OPTIMAL:
        routes: List[List[int]] = []
        for start in range(n):
            if start == depot or solver.Value(x[(depot, start)]) == 0:
                continue
            route = [depot, start]
            current = start
            while True:
                next_node = next(
                    k for k in range(n) if k != current and solver.Value(x[(current, k)]) == 1
                )
                route.append(next_node)
                if next_node == depot:
                    break
                current = next_node
            routes.append(route)
        return routes
    return []