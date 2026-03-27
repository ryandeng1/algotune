from typing import List
from ortools.sat.python import cp_model

def solve(problem: List[List[int]]) -> List[int]:
    """Max independent set via CP-SAT with adjacency list preprocessing."""
    n = len(problem)
    if n == 0:
        return []

    # Build edge list from upper triangular part
    edges = [(i, j)
             for i in range(n)
             for j in range(i + 1, n)
             if problem[i][j]]

    model = cp_model.CpModel()
    vars_ = [model.NewBoolVar(f"x{i}") for i in range(n)]

    # Add constraints: at most one endpoint of each edge in the set
    for i, j in edges:
        model.Add(vars_[i] + vars_[j] <= 1)

    # Objective: maximize the number of selected vertices
    model.Maximize(sum(vars_))

    # Solve
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    # Return nodes with value 1 if a solution is found
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return [i for i, v in enumerate(vars_) if solver.Value(v)]
    return []