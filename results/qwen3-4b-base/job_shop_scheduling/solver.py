import random
from typing import Any
from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem: dict[str, Any]) -> list[list[int]]:
        M = problem["num_machines"]
        jobs_data = problem["jobs"]
        
        job_sums = [sum(p for _, p in job) for job in jobs_data]
        horizon = max(job_sums) + 1 if job_sums else 0
        
        model = cp_model.CpModel()
        
        machine_to_intervals = {m: [] for m in range(M)}
        all_tasks = {}
        
        for j, job in enumerate(jobs_data):
            for k, (m, p) in enumerate(job):
                suffix = f"_{j}_{k}"
                start = model.NewIntVar(0, horizon, f"start_{suffix}")
                end = model.NewIntVar(0, horizon, f"end_{suffix}")
                interval = model.NewIntervalVar(start, p, end, f"interval_{suffix}")
                all_tasks[(j, k)] = (start, end, p)
                machine_to_intervals[m].append(interval)
                if k > 0:
                    prev_end = all_tasks[(j, k - 1)][1]
                    model.Add(start >= prev_end)
        
        for m in range(M):
            model.AddNoOverlap(machine_to_intervals[m])
        
        last_ends = []
        for j, job in enumerate(jobs_data):
            last_task = (j, len(job) - 1)
            last_end = all_tasks[last_task][1]
            last_ends.append(last_end)
        
        makespan = model.NewIntVar(0, horizon, "makespan")
        model.AddMaxEquality(makespan, last_ends)
        model.Minimize(makespan)
        
        solver = cp_model.CpSolver()
        status = solver.Solve(model)
        
        if status == cp_model.OPTIMAL:
            solution = []
            for j, job in enumerate(jobs_data):
                starts = []
                for k, _ in enumerate(job):
                    starts.append(int(solver.Value(all_tasks[(j, k)][0])))
                solution.append(starts)
            return solution
        else:
            return []
