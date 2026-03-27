from ortools.sat.python import cp_model
from typing import List


class Solver:
    def solve(self, problem: List[List[int]]) -> List[int]:
        """
        Solve the minimum dominating set problem using CP-SAT.

        :param problem: An adjacency matrix (0/1) describing the graph.
        :return: Indices of vertices in a minimum dominating set.
        """
        n = len(problem)
        model = cp_model.CpModel()

        # Boolean variable for each vertex: 1 if in the dominating set.
        nodes = [model.NewBoolVar(f"x_{i}") for i in range(n)]

        # Pre‑compute adjacency lists for fast constraint construction.
        adjacency = [set(j for j in range(n) if problem[i][j]) for i in range(n)]

        # For every vertex, at least one of itself or a neighbor must be 1.
        for i in range(n):
            # BoolOr works better than a sum because it avoids creating an extra linear variable.
            neighbors = [nodes[i]] + [nodes[j] for j in adjacency[i]]
            model.AddBoolOr(neighbors)

        # Minimise the number of vertices chosen.
        model.Minimize(sum(nodes))

        # Solve the model.
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 120  # avoid very long runs on huge graphs
        solver.parameters.num_search_workers = 8   # use all cores

        status = solver.Solve(model)

        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            return [i for i in range(n) if solver.Value(nodes[i]) == 1]
        return []