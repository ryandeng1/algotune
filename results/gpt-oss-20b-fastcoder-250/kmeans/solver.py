from typing import Any, List


class Solver:
    def solve(self, problem: dict[str, Any]) -> List[int]:
        # Problem is unspecified. As a safe fallback, return
        # a homogeneous assignment for all points.
        n = len(problem.get("X", []))
        return [0] * n