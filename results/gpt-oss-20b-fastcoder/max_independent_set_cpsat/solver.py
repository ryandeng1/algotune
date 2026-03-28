from typing import List
from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem: List[List[int]]) -> List[int]:
        """
        Solves the maximum independent set problem using OR-Tools CP-SAT.

        :param problem: 2D adjacency matrix (list of lists of 0/1).
        :return: List of node indices that form a maximum independent set.
        """
        n = len(problem)
        model = cp_model.CpModel()
        # Boolean variable for each node: 1 if in the set, 0 otherwise
        nodes = [model.NewBoolVar(f'x_{i}') for i in range(n)]

        # Add constraints: for every edge, at most one endpoint is selected
        for i in range(n):
            for j in range(i + 1, n):
                if problem[i][j]:
                    model.Add(nodes[i] + nodes[j] <= 1)

        # Maximize the number of selected nodes
        model.Maximize(sum(nodes))

        solver = cp_model.CpSolver()
        status = solver.Solve(model)

        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            return [i for i in range(n) if solver.Value(nodes[i]) == 1]
        return []