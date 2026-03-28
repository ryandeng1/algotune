from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem: list[list[int]]) -> list[int]:
        n = len(problem)
        model = cp_model.CpModel()
        x = [model.NewBoolVar(f"x_{i}") for i in range(n)]

        # Add conflicts only once
        add = model.Add
        for i in range(n):
            pi = problem[i]
            xi = x[i]
            for j in range(i + 1, n):
                if pi[j]:          # graph is undirected, adjacency is symmetric
                    add(xi + x[j] <= 1)

        # Objective: maximize the number of selected nodes
        model.Maximize(sum(x))

        solver = cp_model.CpSolver()
        if solver.Solve(model) == cp_model.OPTIMAL:
            return [i for i, var in enumerate(x) if solver.Value(var)]
        return []