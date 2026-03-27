from typing import Any, Dict, List, Tuple

class Result:
    def __init__(self, success: bool, y: List[List[float]], message: str = ""):
        self.success = success
        self.y = y
        self.message = message

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[float]]:
        # Problem is expected to contain:
        #   * "y0": List[float] - initial state
        #   * "dt": float - time step
        #   * "steps": int - number of integration steps
        #   * "f": function(state, t) -> List[float] - derivative function
        try:
            y0: List[float] = problem["y0"]
            dt: float = problem["dt"]
            steps: int = problem["steps"]
            f = problem["f"]
        except KeyError as exc:
            raise ValueError(f"Missing required problem field: {exc}") from exc

        n = len(y0)
        y = [[0.0] * (steps + 1) for _ in range(n)]
        for i in range(n):
            y[i][0] = y0[i]

        t = 0.0
        for step in range(1, steps + 1):
            dydt = f(y, t)
            if len(dydt) != n:
                raise ValueError(
                    f"Derivative function returned {len(dydt)} values, expected {n}"
                )
            for i in range(n):
                y[i][step] = y[i][step - 1] + dt * dydt[i]
            t += dt

        return Result(True, y)

    def _solve(self, problem: Dict[str, Any], debug: bool = False) -> Result:
        return self.solve(problem)

    def wrapper(self, problem: Dict[str, Any]) -> Dict[str, List[float]]:
        sol = self._solve(problem, debug=False)
        if sol.success:
            # Return the last column of y (the final state)
            return [col[-1] for col in sol.y]
        else:
            raise RuntimeError(f"Solver failed: {sol.message}")