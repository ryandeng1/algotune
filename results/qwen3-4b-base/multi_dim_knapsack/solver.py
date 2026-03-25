import numpy as np
from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem):
        value, demand, supply = problem
        n = len(value)
        k = len(supply)
        
        if n <= 20:
            demand_np = np.array(demand)
            value_np = np.array(value)
            
            best_value = -1
            best_indices = []
            
            for mask in range(1 << n):
                selected = np.where((mask >> np.arange(n)) & 1)[0]
                if len(selected) == 0:
                    continue
                total_value = np.sum(value_np[selected])
                usage = np.zeros(k)
                for i in selected:
                    usage += demand_np[i]
                valid = True
                for r in range(k):
                    if usage[r] > supply[r]:
                        valid = False
                        break
                if valid and total_value > best_value:
                    best_value = total_value
                    best_indices = selected.tolist()
            return best_indices
        else:
            model = cp_model.CpModel()
            x = [model.NewBoolVar(f"x_{i}") for i in range(n)]
            
            for r in range(k):
                model.Add(sum(x[i] * demand[i][r] for i in range(n)) <= supply[r])
            model.Maximize(sum(x[i] * value[i] for i in range(n)))
            
            solver = cp_model.CpSolver()
            status = solver.Solve(model)
            
            if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                return [i for i in range(n) if solver.Value(x[i])]
            return []
