import numpy as np
from ortools.sat.python import cp_model


def _preprocess_attacks(board: np.ndarray):
    """
    For each free cell compute the list of cells that are in attack range.
    """
    n, m = board.shape
    directions = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),           (0, 1),
                  (1, -1),  (1, 0),  (1, 1)]
    attack_map = [[[] for _ in range(m)] for _ in range(n)]

    for r in range(n):
        for c in range(m):
            if board[r, c]:
                continue
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                while 0 <= nr < n and 0 <= nc < m:
                    if board[nr, nc]:
                        break
                    attack_map[r][c].append((nr, nc))
                    nr += dr
                    nc += dc
    return attack_map


class Solver:
    def solve(self, problem: np.ndarray) -> list[tuple[int, int]]:
        """
        Solve the Queens with Obstacles Problem.

        Parameters:
            problem (np.ndarray): Boolean matrix denoting obstacles (True if obstacle).

        Returns:
            list[tuple[int, int]]: coordinates of the placed queens.
        """
        n, m = problem.shape
        attack_map = _preprocess_attacks(problem)

        model = cp_model.CpModel()
        # Create a matrix of BoolVars
        vars = [[model.NewBoolVar(f"q_{r}_{c}") for c in range(m)] for r in range(n)]

        # No queen on an obstacle
        for r in range(n):
            for c in range(m):
                if not problem[r, c]:
                    continue
                model.Add(vars[r][c] == 0)

        # Attack constraints
        for r in range(n):
            for c in range(m):
                if problem[r, c]:
                    continue
                if not attack_map[r][c]:
                    continue
                # If a queen is placed at (r,c), all reachable cells must be empty
                attack_vars = [vars[nr][nc] for nr, nc in attack_map[r][c]]
                model.Add(sum(attack_vars) == 0).OnlyEnforceIf(vars[r][c])

        # Objective: maximize number of queens
        all_vars = [vars[r][c] for r in range(n) for c in range(m) if not problem[r, c]]
        model.Maximize(sum(all_vars))

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 60.0  # allow a reasonable time limit
        solver.parameters.random_seed = 42
        status = solver.Solve(model)

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return [(r, c) for r in range(n) for c in range(m)
                    if solver.Value(vars[r][c])]
        return []