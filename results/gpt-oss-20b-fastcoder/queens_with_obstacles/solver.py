import numpy as np
from ortools.sat.python import cp_model

def solver(problem: np.ndarray) -> list[tuple[int, int]]:
    """
    Solve the maximum independent set on the chessboard graph where edges connect
    two cells on the same row, column or diagonal without an obstacle between them.

    Parameters
    ----------
    problem : np.ndarray
        Boolean grid: 0 for empty, 1 for obstacle.

    Returns
    -------
    list[tuple[int, int]]
        Positions of queens in an optimal placement.
    """
    n, m = problem.shape
    model = cp_model.CpModel()

    # Boolean variables for each cell
    queens = [[model.NewBoolVar(f'q_{r}_{c}') for c in range(m)] for r in range(n)]

    # Impose zero on obstacle cells
    for r in range(n):
        for c in range(m):
            if problem[r, c]:
                model.Add(queens[r][c] == 0)

    # Helper to add "at most one" constraints along segments
    def add_at_most_one(indices):
        if indices:
            vars_ = [queens[i] for i in indices]
            model.Add(sum(vars_) <= 1)

    # Rows
    for r in range(n):
        seg = []
        for c in range(m):
            if problem[r, c]:
                add_at_most_one(seg)
                seg = []
            else:
                seg.append((r, c))
        add_at_most_one(seg)

    # Columns
    for c in range(m):
        seg = []
        for r in range(n):
            if problem[r, c]:
                add_at_most_one(seg)
                seg = []
            else:
                seg.append((r, c))
        add_at_most_one(seg)

    # Diagonals (top-left to bottom-right)
    for k in range(-n + 1, m):
        seg = []
        for r in range(n):
            c = r + k
            if 0 <= c < m:
                if problem[r, c]:
                    add_at_most_one(seg)
                    seg = []
                else:
                    seg.append((r, c))
        add_at_most_one(seg)

    # Anti-diagonals (top-right to bottom-left)
    for k in range(n + m - 1):
        seg = []
        for r in range(n):
            c = k - r
            if 0 <= c < m:
                if problem[r, c]:
                    add_at_most_one(seg)
                    seg = []
                else:
                    seg.append((r, c))
        add_at_most_one(seg)

    # Objective: maximize number of queens
    model.Maximize(sum(queens[r][c] for r in range(n) for c in range(m)))

    # Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 30.0
    status = solver.Solve(model)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return [(r, c) for r in range(n) for c in range(m) if solver.Value(queens[r][c])]
    return []

# Example usage:  solve(np.array([[0,0,1],[0,0,0],[1,0,0]]))