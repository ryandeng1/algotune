# solver.py
from typing import List
from ortools.sat.python import cp_model

class Solver:
    """
    Optimized CP‑SAT solution for the minimum dominating set problem.

    The implementation focuses on avoiding Python overhead in the tight loops.
    All graph information is pre‑processed into adjacency lists before the
    constraints are added to the model, reducing the number of nested
    Python loops at runtime.
    """

    def solve(self, problem: List[List[int]]) -> List[int]:
        """
        Finds a minimum dominating set for the given graph.

        :param problem: Adjacency matrix (list of lists) with 1 for an edge
                        and 0 otherwise.
        :return: List of vertex indices that form a minimum dominating set.
        """
        # Number of vertices
        n = len(problem)

        # Pre‑compute the adjacency list for each vertex.
        # Each entry contains the index of the vertex itself and all
        # neighbours (so the constraint is that the sum of the
        # corresponding BoolVars is at least one).
        neighbors_of = []
        for i, row in enumerate(problem):
            # Use list comprehension for speed
            neigh = [i]
            neigh.extend(j for j, val in enumerate(row) if val)
            neighbors_of.append(neigh)

        # Create the CP‑SAT model
        model = cp_model.CpModel()
        nodes = [model.NewBoolVar(f'x_{i}') for i in range(n)]

        # Add dominating constraints
        # Note: sum() on a list of BoolVars is efficient in CP‑SAT.
        for neigh in neighbors_of:
            model.Add(sum(nodes[j] for j in neigh) >= 1)

        # Minimise the number of selected vertices
        model.Minimize(sum(nodes))

        # Create solver and solve
        solver = cp_model.CpSolver()

        # The default solver parameters are usually fast enough.
        # For very large graphs you could tune time limits, threads, etc.
        status = solver.Solve(model)

        # Extract solution
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            return [i for i, var in enumerate(nodes) if solver.Value(var)]
        return []