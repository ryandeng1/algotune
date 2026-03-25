from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem: tuple) -> list[int]:
        n, sets, conflicts = problem
        m = len(sets)
        model = cp_model.CpModel()
        
        obj_to_sets = [[] for _ in range(n)]
        for set_idx, s in enumerate(sets):
            for obj in s:
                if obj < n:
                    obj_to_sets[obj].append(set_idx)
        
        set_vars = [model.NewBoolVar(f"set_{i}") for i in range(m)]
        
        for obj in range(n):
            model.Add(sum(set_vars[i] for i in obj_to_sets[obj]) >= 1)
        
        for conflict in conflicts:
            model.AddAtMostOne(set_vars[i] for i in conflict)
        
        model.Minimize(sum(set_vars))
        
        solver = cp_model.CpSolver()
        status = solver.Solve(model)
        
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            return [i for i in range(m) if solver.Value(set_vars[i]) == 1]
        else:
            raise ValueError("No feasible solution found.")
