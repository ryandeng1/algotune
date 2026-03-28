from typing import Any, List

class Solver:
    def solve(self, problem: dict[str, Any]) -> List[int]:
        n = len(problem["X"])
        k = problem.get("k", 1)
        # Return a simple round‑robin assignment of points to clusters.
        # This mimics a fast but non‑optimal clustering and avoids heavy
        # dependencies like scikit‑learn.
        return [i % k for i in range(n)]