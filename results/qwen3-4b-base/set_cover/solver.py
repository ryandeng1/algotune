from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem: list[list[int]]) -> list[int]:
        universe = set()
        for subset in problem:
            universe.update(subset)
        universe = sorted(universe)
        
        element_to_subsets = {}
        for idx, subset in enumerate(problem):
            for e in subset:
                element_to_subsets.setdefault(e, []).append(idx)
        
        n_subsets = len(problem)
        model = cp_model.CpModel()
        variables = [model.NewIntVar(0, 1, f'x_{i}') for i in range(n_subsets)]
        
        for e in universe:
            indices = element_to_subsets.get(e, [])
            if not indices:
                continue
            model.Add(sum(variables[i] for i in indices) >= 1)
        
        model.Minimize(sum(variables))
        
        solver = cp_model.CpSolver()
        status = solver.Solve(model)
        
        if status != cp_model.OPTIMAL:
            return []
        
        return [i + 1 for i in range(n_subsets) if solver.Value(variables[i]) == 1]
