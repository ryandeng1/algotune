from ortools.sat.python import cp_model

def solve(problem: dict) -> list[list[int]]:
    D, K, depot = problem["D"], problem["K"], problem["depot"]
    n = len(D)

    model = cp_model.CpModel()

    # Binary variables: x[i][j] == 1 if edge i->j is used
    x = {(i, j): model.NewBoolVar(f"x_{i}_{j}") for i in range(n) for j in range(n) if i != j}

    # Degree constraints for non‑depot nodes
    for i in range(n):
        if i == depot:
            continue
        model.Add(sum(x[j, i] for j in range(n) if j != i) == 1)
        model.Add(sum(x[i, j] for j in range(n) if j != i) == 1)

    # Depot constraints
    model.Add(sum(x[depot, j] for j in range(n) if j != depot) == K)
    model.Add(sum(x[i, depot] for i in range(n) if i != depot) == K)

    # Subtour elimination (Miller‑Hobbs‑Tucker)
    u = {i: model.NewIntVar(1, n - 1, f"u_{i}") for i in range(n) if i != depot}
    for i in range(n):
        if i == depot:
            continue
        for j in range(n):
            if j == depot or i == j:
                continue
            model.Add(u[i] + 1 <= u[j] + (n - 1) * (1 - x[i, j]))

    # Objective: minimize total distance
    model.Minimize(sum(D[i][j] * x[i, j] for i, j in x))

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status != cp_model.OPTIMAL:
        return []

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