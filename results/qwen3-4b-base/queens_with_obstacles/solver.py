import numpy as np
from ortools.sat.python import cp_model
from numba import njit

@njit
def get_reach_positions(board, r, c):
    n, m = board.shape
    directions = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 1),
        (1, -1), (1, 0), (1, 1)
    ]
    positions = []
    for dr, dc in directions:
        nr, nc = r + dr, c + dc
        while 0 <= nr < n and 0 <= nc < m:
            if board[nr, nc]:
                break
            positions.append((nr, nc))
            nr += dr
            nc += dc
    return positions

class Solver:
    def solve(self, problem: np.ndarray) -> list[tuple[int, int]]:
        instance = problem
        n, m = instance.shape
        model = cp_model.CpModel()
        
        reach_positions = [[None] * m for _ in range(n)]
        for r in range(n):
            for c in range(m):
                if not instance[r, c]:
                    reach_positions[r][c] = get_reach_positions(instance, r, c)
                else:
                    reach_positions[r][c] = []
        
        queens = [[model.NewBoolVar(f"queen_{r}_{c}") for c in range(m)] for r in range(n)]
        
        for r in range(n):
            for c in range(m):
                if instance[r, c]:
                    model.Add(queens[r][c] == 0)
        
        for r in range(n):
            for c in range(m):
                if not instance[r, c]:
                    model.Add(
                        sum(queens[nr][nc] for nr, nc in reach_positions[r][c]) == 0
                    ).OnlyEnforceIf(queens[r][c])
        
        model.Maximize(sum(queens[r][c] for r in range(n) for c in range(m)))
        
        solver = cp_model.CpSolver()
        solver.parameters.log_search_progress = False
        status = solver.Solve(model)
        
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            return [(r, c) for r in range(n) for c in range(m) if solver.Value(queens[r][c])]
        else:
            return []
