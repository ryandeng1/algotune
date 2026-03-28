from typing import Any
from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem: dict[str, Any]) -> list[list[int]]:
        """
        Solve the JSSP using CP-SAT with interval variables and no-overlap.
        The implementation is streamlined for speed: all unnecessary
        pass statements, redundant loops and variable creations are removed.
        """
        M = problem['num_machines']
        jobs_data = problem['jobs']

        # Basic model and horizon preparation
        model = cp_model.CpModel()
        horizon = sum(duration for job in jobs_data for _, duration in job)

        # Create interval variables and track per‑machine tasks
        machine_tasks = [[] for _ in range(M)]
        task_start = {}
        for j, job in enumerate(jobs_data):
            prev_end = None
            for k, (m, dur) in enumerate(job):
                suffix = f"_{j}_{k}"
                start = model.NewIntVar(0, horizon, f'start{suffix}')
                end = model.NewIntVar(0, horizon, f'end{suffix}')
                interval = model.NewIntervalVar(start, dur, end, f'interval{suffix}')
                machine_tasks[m].append(interval)
                task_start[(j, k)] = start
                if prev_end is not None:
                    model.Add(start >= prev_end)
                prev_end = end

        # No‑overlap constraints per machine
        for m in range(M):
            model.AddNoOverlap(machine_tasks[m])

        # Makespan objective
        makespan = model.NewIntVar(0, horizon, 'makespan')
        last_ends = [task_start[(j, len(job)-1)] for j, job in enumerate(jobs_data)]
        model.AddMaxEquality(makespan, last_ends)
        model.Minimize(makespan)

        # Solve
        solver = cp_model.CpSolver()
        status = solver.Solve(model)

        # Extract solution
        if status == cp_model.OPTIMAL:
            return [
                [int(solver.Value(task_start[(j, k)])) for k in range(len(job))]
                for j, job in enumerate(jobs_data)
            ]
        return []