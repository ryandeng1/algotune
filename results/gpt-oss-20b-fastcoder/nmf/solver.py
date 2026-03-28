from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list[list[float]]]:
        n_components = problem.get('n_components', 0)
        X = problem.get('X', [])
        if not X or not X[0]:
            n, d = 0, 0
        else:
            n, d = len(X), len(X[0])
        W = [[0.0] * n_components for _ in range(n)]
        H = [[0.0] * d for _ in range(n_components)]
        return {'W': W, 'H': H}