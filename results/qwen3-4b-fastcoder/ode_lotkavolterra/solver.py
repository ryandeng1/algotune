def solve(self, problem: dict[str, np.ndarray | float]) -> dict[str, list[float]]:
    sol = self._solve(problem, debug=False)
    if sol.success:
        return sol.y[:, -1].tolist()
    else:
        raise RuntimeError(f"Solver failed: {sol.message}")