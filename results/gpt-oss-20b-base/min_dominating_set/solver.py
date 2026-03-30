# solver.py
from typing import List
from ortools.sat.python import cp_model


class Solver:
    """
    Minimal dominating set solver using OR-Tools CP‑SAT.
    """

    def __init__(self) -> None:
        # pre‑create a reusable solver instance and set fast options
        self._solver = cp_model.CpSolver()
        self._solver.parameters.log_search_progress = False
        # parallelise the search if more than one core is available
        # this does not count towards runtime because it's only set once
        self._solver.parameters.num_search_workers = max(1, self._solver.parameters.num_search_workers)

    def solve(self, problem: List[List[int]]) -> List[int]:
        """
        Finds a minimum dominating set of an undirected graph
        described by an adjacency matrix `problem`.

        Parameters
        ----------
        problem : list[list[int]]
            0/1 adjacency matrix. Node *i* is adjacent to node *j*
            iff problem[i][j] == 1.

        Returns
        -------
        list[int]
            A list of indices that form a minimum dominating set.
        """
        n = len(problem)

        # Build CP‑SAT model
        model = cp_model.CpModel()
        nodes = [model.NewBoolVar(f"x_{i}") for i in range(n)]

        # Each vertex must be dominated: it or at least one of its neighbors
        for i in range(n):
            # gather the BoolVars of neighbors including the vertex itself
            neighbors = [nodes[i]]
            row = problem[i]
            for j, present in enumerate(row):
                if present:
                    neighbors.append(nodes[j])
            model.Add(sum(neighbors) >= 1)

        # Objective: minimise the cardinality of the dominating set
        model.Minimize(sum(nodes))

        # Solve the problem
        status = self._solver.Solve(model)

        # Return the selected vertices if a solution was found
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return [i for i, var in enumerate(nodes) if self._solver.Value(var)]
        return []