from ortools.sat.python import cp_model

def solve(problem):
    """
    optimal multi‑dimensional knapsack

    Returns a list of selected item indices; empty list on failure.
    """
    # Unpack the instance
    if isinstance(problem, tuple) or isinstance(problem, list):
        value, demand, supply = problem
    else:
        value, demand, supply = problem.value, problem.demand, problem.supply

    n, k = len(value), len(supply)

    # Build model
    model = cp_model.CpModel()
    x = [model.NewBoolVar(f'x_{i}') for i in range(n)]

    for r in range(k):
        model.Add(sum(x[i] * demand[i][r] for i in range(n)) <= supply[r])

    model.Maximize(sum(x[i] * value[i] for i in range(n)))

    # Solve
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return [i for i in range(n) if solver.Value(x[i])]
    return []