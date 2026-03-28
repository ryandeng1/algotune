from ortools.sat.python import cp_model
from typing import List, Dict

class Solver:
    def solve(self, problem: Dict[str, List]) -> List[int]:
        """
        Solves the Maximum Weight Independent Set (MWIS) problem efficiently
        using OR-Tools CP-SAT.

        Parameters
        ----------
        problem : dict
            A dictionary containing:
                'adj_matrix': List[List[int]] – adjacency matrix (0/1)
                'weights'   : List[int]      – node weights

        Returns
        -------
        List[int]
            A list of node indices that form the optimal MWIS.
        """
        adj_matrix = problem["adj_matrix"]
        weights = problem["weights"]
        n = len(adj_matrix)

        model = cp_model.CpModel()
        nodes = [model.NewBoolVar(f"x_{i}") for i in range(n)]

        # Add conflict constraints only for existing edges
        for i in range(n):
            row = adj_matrix[i]
            for j in range(i + 1, n):
                if row[j]:
                    model.Add(nodes[i] + nodes[j] <= 1)

        # Optimize the sum of selected weights
        model.Maximize(sum(weights[i] * nodes[i] for i in range(n)))

        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = max(1, cp_model.DefaultSearchWorkers())
        status = solver.Solve(model)

        if status == cp_model.OPTIMAL:
            return [i for i in range(n) if solver.Value(nodes[i])]
        return []