class Solver:
    def solve(self, problem: dict[str, np.ndarray | float]) -> dict[str, list[float]]:
        sol = self._solve(problem, debug=False)
        if sol.success:
            final_values = sol.y[:, -1].tolist()
            return {k: [v] for k, v in zip(problem.keys(), final_values)}
        else:
            raise RuntimeError(f"Solver failed: {sol.message}")