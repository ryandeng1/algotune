# solver.py

import itertools
from ortools.sat.python import cp_model
from typing import Any, Dict, List, Tuple


class Solver:
    def solve(self, problem: Dict[str, List[List[int]]], **kwargs) -> List[Tuple[int, int]]:
        """
        Solve the Maximum Common Subgraph problem for two undirected graphs
        represented by their adjacency matrices A and B.

        Parameters
        ----------
        problem : dict
            Dictionary with keys 'A' and 'B', each mapping to a symmetric
            adjacency matrix (list of lists) of 0/1.

        Returns
        -------
        list[tuple[int, int]]
            List of pairs (i, p) indicating that node i in graph G
            corresponds to node p in graph H in an optimal common subgraph.
            The size of the returned list is maximal.
        """
        A = problem["A"]
        B = problem["B"]
        n, m = len(A), len(B)

        # Create CP-SAT model
        model = cp_model.CpModel()

        # Decision variables x[i][p] = 1 iff node i maps to node p
        x = [[model.NewBoolVar(f"x_{i}_{p}") for p in range(m)] for i in range(n)]

        # One-to-one mapping constraints
        for i in range(n):
            model.Add(sum(x[i][p] for p in range(m)) <= 1)
        for p in range(m):
            model.Add(sum(x[i][p] for i in range(n)) <= 1)

        # Edge consistency constraints
        for i in range(n):
            for j in range(i + 1, n):
                for p in range(m):
                    for q in range(p + 1, m):
                        if A[i][j] != B[p][q]:
                            # If mapping incompatible, forbid both assignments
                            model.Add(x[i][p] + x[j][q] <= 1)
                            model.Add(x[i][q] + x[j][p] <= 1)

        # Objective: maximize number of mapped nodes
        model.Maximize(sum(x[i][p] for i in range(n) for p in range(m)))

        # Solve
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = kwargs.get("timeout", 30.0)
        status = solver.Solve(model)

        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            return [(i, p) for i in range(n) for p in range(m) if solver.Value(x[i][p]) == 1]
        else:
            return []
