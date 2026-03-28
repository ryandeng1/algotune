from ortools.sat.python import cp_model
from typing import Any, List

class Solver:
    def solve(self, problem: dict[str, Any]) -> List[List[int]]:
        """Solve Job Shop Scheduling with CP‑SAT."""
        M = problem["num_machines"]
        jobs = problem["jobs"]

        horizon = sum(p for job in jobs for _, p in job)
        model = cp_model.CpModel()

        # Keep intervals per machine for NoOverlap
        machine_intervals = [[] for _ in range(M)]
        # Store start vars for result extraction
        start_vars = {}

        for j, job in enumerate(jobs):
            prev_end = None
            for k, (m, dur) in enumerate(job):
                start = model.NewIntVar(0, horizon, f"s_{j}_{k}")
                end = model.NewIntVar(0, horizon, f"e_{j}_{k}")
                interval = model.NewIntervalVar(start, dur, end, f"i_{j}_{k}")
                machine_intervals[m].append(interval)
                start_vars[(j, k)] = start
                if prev_end is not None:
                    model.Add(start >= prev_end)
                prev_end = end

        for intervals in machine_intervals:
            if intervals:
                model.AddNoOverlap(intervals)

        # Makespan
        makespan = model.NewIntVar(0, horizon, "makespan")
        last_ends = [model.NewIntVar(0, horizon, f"end_{j}") for j in range(len(jobs))]
        for j, job in enumerate(jobs):
            _, end_var, _ = start_vars[(j, len(job) - 1)]
            model.Add(last_ends[j] == end_var)
        model.AddMaxEquality(makespan, last_ends)
        model.Minimize(makespan)

        # Solve
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 30.0  # avoid infinite time
        status = solver.Solve(model)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return []

        # Extract start times
        solution: List[List[int]] = []
        for j, job in enumerate(jobs):
            job_start = [solver.Value(start_vars[(j, k)]) for k in range(len(job))]
            solution.append(job_start)

        return solution