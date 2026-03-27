from ortools.sat.python import cp_model
from typing import List

class Solver:
    def solve(self, problem: List[List[int]]) -> List[int]:
        """
        Solves the maximum clique problem using the CP‑SAT solver.

        :param problem: A 2D adjacency matrix representing the graph.
        :return: A list of node indices that form a maximum clique.
        """
        n = len(problem)
        # Create boolean variables for all nodes
        model = cp_model.CpModel()
        nodes = [model.NewBoolVar(f"x_{i}") for i in range(n)]

        # For every pair of non‑adjacent nodes add: x_i + x_j <= 1
        # Build the list of pairs once to avoid repeated matrix look‑ups
        non_edges = [(i, j) for i in range(n)
                               for j in range(i + 1, n)
                               if problem[i][j] == 0]
        for i, j in non_edges:
            model.Add(nodes[i] + nodes[j] <= 1)

        # Objective: maximize the total number of selected nodes
        model.Maximize(sum(nodes))

        # Solve
        solver = cp_model.CpSolver()
        # Enable a time limit to guard against very large graphs
        solver.parameters.max_time_in_seconds = 60.0
        status = solver.Solve(model)

        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            return [i for i in range(n) if solver.Value(nodes[i]) == 1]
        return []