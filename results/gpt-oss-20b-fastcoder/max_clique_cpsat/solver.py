from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem: list[list[int]]) -> list[int]:
        """
        Solves the maximum clique problem using the CP‑SAT solver.

        :param problem: A 2D adjacency matrix representing the graph.
        :return: A list of node indices that form a maximum clique.
        """
        n = len(problem)
        model = cp_model.CpModel()
        nodes = [model.NewBoolVar(f'x_{i}') for i in range(n)]

        # Add constraints only for non‑edges
        for i in range(n):
            pi = problem[i]
            ni = nodes[i]
            for j in range(i + 1, n):
                if pi[j] == 0:
                    model.Add(ni + nodes[j] <= 1)

        model.Maximize(sum(nodes))
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 30.0  # optional cap

        status = solver.Solve(model)
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            return [i for i, v in enumerate(nodes) if solver.Value(v)]
        return []