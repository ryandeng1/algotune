from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem: list[list[int]]) -> list[int]:
        """Finds a minimum dominating set of a graph given by its adjacency matrix."""
        n = len(problem)
        model = cp_model.CpModel()
        nodes = [model.NewBoolVar(f"x{i}") for i in range(n)]

        # Add constraints: each node must be dominated
        for i in range(n):
            # A node dominates itself and its neighbors
            neighbors = [nodes[j] for j, val in enumerate(problem[i]) if val]
            neighbors.append(nodes[i])           # itself
            model.Add(sum(neighbors) >= 1)

        # Objective: minimize number of selected nodes
        model.Minimize(sum(nodes))
        solver = cp_model.CpSolver()
        status = solver.Solve(model)

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return [i for i, v in enumerate(nodes) if solver.Value(v) == 1]
        return []