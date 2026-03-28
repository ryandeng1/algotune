from typing import List
import numpy as np
from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem: np.ndarray) -> list[tuple[int, int]]:
        n, m = problem.shape
        model = cp_model.CpModel()
        queens = [[model.NewBoolVar(f"queen_{i}_{j}") for j in range(m)] for i in range(n)]
        
        directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1), (0, 1),
            (1, -1), (1, 0), (1, 1)
        ]
        
        reach_maps = [[None] * m for _ in range(n)]
        for r in range(n):
            for c in range(m):
                if problem[r, c]:
                    continue
                reachable = []
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    while 0 <= nr < n and 0 <= nc < m:
                        if problem[nr, nc]:
                            break
                        reachable.append((nr, nc))
                        nr += dr
                        nc += dc
                reach_maps[r][c] = reachable
        
        for r in range(n):
            for c in range(m):
                if problem[r, c]:
                    continue
                reachable = reach_maps[r][c]
                model.Add(sum(queens[nr][nc] for nr, nc in reachable) == 0).OnlyEnforceIf(queens[r][c])
        
        model.Maximize(sum(queens[r][c] for r in range(n) for c in range(m)))
        
        solver = cp_model.CpSolver()
        solver.parameters.log_search_progress = True
        status = solver.Solve(model)
        
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return [(r, c) for r in range(n) for c in range(m) if solver.Value(queens[r][c])]
        else:
            return []