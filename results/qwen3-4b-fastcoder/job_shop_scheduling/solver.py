from typing import Any
from ortools.sat.python import cp_model


class Solver:
    def solve(self, problem: dict[str, Any]) -> list[list[int]]:
        M = problem["num_machines"]
        jobs_data = problem["jobs"]
        horizon = sum(d for job in jobs_data for _, d in job)

        model = cp_model.CpModel()
        all_tasks = {}  # (j,k) -> (start, interval)
        machine_to_intervals = {m: [] for m in range(M)}

        for j, job in enumerate(jobs_data):
            for k, (m, p) in enumerate(job):
                suffix = f"_{j}_{k}"
                start = model.NewIntVar(0, horizon, f"start{suffix}")
                interval = model.NewIntervalVar(start, p, f"interval{suffix}")
                all_tasks[(j, k)] = (start, interval)
                machine_to_intervals[m].append(interval)
                if k > 0:
                    prev_start, prev_interval = all_tasks[(j, k - 1)]
                    model.Add(start >= prev_interval.End())

        for m in range(M):
            model.AddNoOverlap(machine_to_intervals[m])

        makespan = model.NewIntVar(0, horizon, "makespan")
        last_ends = []
        for j, job in enumerate(jobs_data):
            _, last_interval = all_tasks[(j, len(job) - 1)]
            last_ends.append(last_interval.End())
        model.AddMaxEquality(makespan, last_ends)
        model.Minimize(makespan)

        solver = cp_model.CpSolver()
        status = solver.Solve(model)

        if status == cp_model.OPTIMAL:
            solution = []
            for j, job in enumerate(jobs_data):
                starts = [int(solver.Value(all_tasks[(j, k)][0])) for k in range(len(job))]
                solution.append(starts)
            return solution
        else:
            return []