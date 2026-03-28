from ortools.sat.python import cp_model

def solve(problem: dict[str, list[list[int]]]) -> list[tuple[int, int]]:
    A, B = problem["A"], problem["B"]
    n, m = len(A), len(B)
    model = cp_model.CpModel()
    # Decision variables: x[i][p] = 1 if A index i is mapped to B index p
    x = [[model.NewBoolVar(f"x_{i}_{p}") for p in range(m)] for i in range(n)]

    # Each A index is used at most once
    for i in range(n):
        model.Add(sum(x[i]) <= 1)
    # Each B index is used at most once
    for p in range(m):
        model.Add(sum(x[i][p] for i in range(n)) <= 1)

    # Build a quick lookup for B values
    B_values = {}
    for p in range(m):
        for q in range(m):
            if p != q:
                B_values.setdefault((p, q), B[p][q])

    # Add conflict constraints
    for i in range(n):
        for j in range(i + 1, n):
            a_val = A[i][j]
            for p in range(m):
                for q in range(m):
                    if p == q:
                        continue
                    if a_val != B[p][q]:
                        model.Add(x[i][p] + x[j][q] <= 1)

    model.Maximize(sum(x[i][p] for i in range(n) for p in range(m)))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 5  # optional time limit
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        return [(i, p) for i in range(n) for p in range(m) if solver.Value(x[i][p])]
    return []