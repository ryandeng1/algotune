from typing import List, Final
from ortools.sat.python import cp_model

# -------------------------------------------------------------

class Solver:
    """
    Maximize an independent set using CP‑SAT.
    Optimized to build a minimal model and to use efficient data structures.
    """

    # CP‑SAT keeps internal limits; raising the timeout can help for larger graphs.
    _DEFAULT_TIMEOUT: Final[int] = 300  # seconds

    def solve(self, problem: List[List[int]]) -> List[int]:
        n = len(problem)

        # Quick elimination: if graph has no edges, return all vertices.
        if not any(any(row) for row in problem):
            return list(range(n))

        # Helper: gather all edges from the adj‑matrix.
        edges = [(i, j) for i in range(n) for j in range(i + 1, n) if problem[i][j]]
        # There is no need to store more than one copy of each edge.

        model = cp_model.CpModel()

        # Boolean vars (0/1) for each vertex: 1 if chosen.
        x = model.NewBoolVarArray(n)

        # Add independency constraints only for edges.
        # Using linear terms is faster than adding many separate constraints.
        for i, j in edges:
            model.Add(x[i] + x[j] <= 1)

        # Maximise number of selected vertices.
        model.Maximize(sum(x))

        solver = cp_model.CpSolver()
        # Lower cost of pre‑processing: disable the default progress output.
        solver.parameters.log_search_progress = False
        solver.parameters.num_search_workers = 0  # single‑thread for deterministic output
        solver.parameters.max_time_in_seconds = self._DEFAULT_TIMEOUT

        status = solver.Solve(model)

        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            return [idx for idx in range(n) if solver.Value(x[idx])]
        return []