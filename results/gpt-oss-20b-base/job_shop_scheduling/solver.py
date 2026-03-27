from typing import Any, List
from ortools.sat.python import cp_model


class Solver:
    def solve(self, problem: dict[str, Any]) -> List[List[int]]:
        """Solve JSSP with CP-SAT."""
        machines = problem["num_machines"]
        jobs = problem["jobs"]

        # Upper bound on the makespan: total processing time
        horizon = sum(d for job in jobs for _, d in job)

        model = cp_model.CpModel()

        # Store start variables and intervals per job and per machine
        starts: List[List[cp_model.IntVar]] = []
        machine_intervals: List[List[cp_model.IntervalVar]] = [[] for _ in range(machines)]

        for j, job in enumerate(jobs):
            job_starts = []
            prev_end = None
            for k, (m, p) in enumerate(job):
                suffix = f"_{j}_{k}"
                start_var = model.NewIntVar(0, horizon, f"start{suffix}")
                job_starts.append(start_var)
                interval = model.NewIntervalVar(start_var, p, start_var + p, f"interval{suffix}")
                machine_intervals[m].append(interval)
                if prev_end is not None:
                    model.Add(start_var >= prev_end)
                prev_end = start_var + p
            starts.append(job_starts)

        # No‑overlap constraints per machine
        for m in range(machines):
            model.AddNoOverlap(machine_intervals[m])

        # Makespan minimization
        makespan = model.NewIntVar(0, horizon, "makespan")
        end_vars = [starts[j][len(jobs[j]) - 1] + sum(d for _, d in jobs[j]) for j in range(len(jobs))]
        model.AddMaxEquality(makespan, end_vars)
        model.Minimize(makespan)

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 30  # optional time limit
        status = solver.Solve(model)

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return [[solver.Value(s) for s in job_starts] for job_starts in starts]
        return []