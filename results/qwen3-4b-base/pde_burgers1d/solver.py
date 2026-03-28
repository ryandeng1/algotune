class Solver:
    def solve(self, problem: dict[str, np.ndarray | float]) -> dict[str, list[float]]:
        sol = self._solve(problem, debug=False)
        if sol.success:
            return {k: [float(val)] for k, val in zip(problem.keys(), sol.y[:, -1])}
        else:
            raise RuntimeError(f"Solver failed: {sol.message}")