import random
from typing import Any
from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem: dict[str, Any]) -> list[list[int]]:
        M = problem["num_machines"]
        jobs_data = problem["jobs"]
        
        horizon = sum(d for job in jobs_data for _, d in job)
        model = cp_model.CpModel()
        
        all_tasks = {}
        machine_to_intervals = {m: [] for m in range(M)}
        
        for j, job in enumerate(jobs_data):
            for k, (m, p) in enumerate(job):
                suffix = f"_{j}_{k}"
                start = model.NewIntVar(0, horizon, f"start{suffix}")
                end = model.NewIntVar(0, horizon, f"end{suffix}")
                interval = model.NewIntervalVar(start, p, end, f"interval{suffix}")
                all_tasks[(j, k)] = (start, end, p)
                machine_to_intervals[m].append(interval)
                if k > 0:
                    prev_end = all_tasks[(j, k - 1)][1]
                    model.Add(start >= prev_end)
        
        for m in range(M):
            model.AddNoOverlap(machine_to_intervals[m])
        
        makespan = model.NewIntVar(0, horizon, "makespan")
        last_ends = []
        for job_id, job in enumerate(jobs_data):
            _, end_var, _ = all_tasks[(job_id, len(job) - 1)]
            last_ends.append(end_var)
        model.AddMaxEquality(makespan, last_ends)
        model.Minimize(makespan)
        
        solver = cp_model.CpSolver()
        solver.parameters.search_strategy = 'LB'
        solver.parameters.num_threads = 4
        status = solver.Solve(model)
        
        if status == cp_model.OPTIMAL:
            solution = []
            for j, job in enumerate(jobs_data):
                starts = []
                for k in range(len(job)):
                    starts.append(int(solver.Value(all_tasks[(j, k)][0])))
                solution.append(starts)
            return solution
        else:
            return []
