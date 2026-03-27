from typing import List, Dict
from ortools.sat.python import cp_model


class Solver:
    def solve(self, problem: Dict[str, List]) -> List[int]:
        """Solve the weighted maximum independent set (MWIS) problem using
        OR-Tools CP‑SAT in an efficient way.

        Parameters
        ----------
        problem : dict
            Must contain the keys ``"adj_matrix"`` and ``"weights"``.
            * ``"adj_matrix"`` – a square list of lists with ``True`` on
              edges and ``False`` otherwise.
            * ``"weights"`` – a list of non‑negative integers, one per node.

        Returns
        -------
        list[int]
            Indices of the vertices that belong to an optimal independent set.
        """
        # --- Build a compact representation of the adjacency structure. ---
        adj_matrix = problem["adj_matrix"]
        weights = problem["weights"]
        n = len(adj_matrix)

        # Convert adjacency matrix to adjacency lists to avoid generating
        # constraints for non‑adjacent pairs.
        neighbor_lists = [[] for _ in range(n)]
        for i in range(n):
            row = adj_matrix[i]
            for j in range(i + 1, n):
                if row[j]:
                    neighbor_lists[i].append(j)
                    neighbor_lists[j].append(i)

        # --- Construct the CP‑SAT model. ---
        model = cp_model.CpModel()
        nodes = [model.NewBoolVar(f"x_{i}") for i in range(n)]

        # For every edge (i, j) add the constraint x_i + x_j <= 1.
        for i in range(n):
            for j in neighbor_lists[i]:
                if i < j:        # each undirected edge only once
                    model.Add(nodes[i] + nodes[j] <= 1)

        # Objective: maximize the weighted sum of selected nodes.
        model.Maximize(
            sum(weights[i] * nodes[i] for i in range(n))
        )

        # --- Solve the model. ---
        solver = cp_model.CpSolver()
        # A small tweak to the search parameters can help most instances.
        solver.parameters.linearization_level = 0
        solver.parameters.max_time_in_seconds = 5.0

        status = solver.Solve(model)

        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            return [i for i in range(n) if solver.Value(nodes[i]) == 1]
        return []