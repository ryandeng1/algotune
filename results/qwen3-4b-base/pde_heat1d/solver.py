class Solver:
    def solve(self, problem: dict[str, np.ndarray | float]) -> dict[str, list[float]]:
        sol = self._solve(problem, debug=False)
        if sol.success:
            final_vals = sol.y[:, -1].tolist()
            return {key: [val] for key, val in zip(problem.keys(), final_vals)}
        else:
            raise RuntimeError(f"Solver failed: {sol.message}")