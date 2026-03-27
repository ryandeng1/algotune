from typing import Any
from ortools.sat.python import cp_model


class Solver:
    def solve(self, problem: dict[str, list]) -> list[int]:
        """
        Solves the MWIS problem using CP‑SAT.

        :param problem: dict with 'adj_matrix' and 'weights'
        :return: list of selected node indices.
        """
        adj = problem["adj_matrix"]
        w = problem["weights"]
        n = len(adj)

        model = cp_model.CpModel()

        # One bool variable per vertex.
        x = [model.NewBoolVar(f"x_{i}") for i in range(n)]

        # Convert adjacency matrix to adjacency lists to avoid O(n²) loops.
        neigh = [set() for _ in range(n)]
        for i in range(n):
            row = adj[i]
            for j, val in enumerate(row):
                if val:
                    neigh[i].add(j)

        # For each edge (i, j) with i < j, add the constraint xi + xj <= 1.
        for i in range(n):
            for j in neigh[i]:
                if j > i:
                    model.Add(x[i] + x[j] <= 1)

        # Maximise total weight.
        objective = sum(w[i] * x[i] for i in range(n))
        model.Maximize(objective)

        solver = cp_model.CpSolver()
        status = solver.Solve(model)

        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            return [i for i in range(n) if solver.Value(x[i])]
        return []