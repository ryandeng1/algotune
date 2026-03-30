from typing import Any, List
from ortools.sat.python import cp_model

# ---------------------------------------------
# Highly optimised CP‑SAT solver for JSSP
# ---------------------------------------------
class Solver:
    def solve(self, problem: dict[str, Any]) -> List[List[int]]:
        """
        Solve a Job‑Shop Scheduling Problem using OR‑Tools CP‑SAT.
        The implementation is deliberately lean: it creates only the
        variables that are absolutely necessary and avoids any
        extraneous loops or temporary objects.
        """
        jobs = problem["jobs"]
        num_machines = problem["num_machines"]

        # Maximum possible makespan – an upper bound that is tight enough
        horizon = sum(p for job in jobs for _, p in job)

        model = cp_model.CpModel()

        # Keep intervals per machine for the NoOverlap constraint
        machine_intervals: List[List[cp_model.IntervalVar]] = [[] for _ in range(num_machines)]

        # For each operation we store its start variable and the duration
        starts: List[List[cp_model.IntVar]] = []

        for j, job in enumerate(jobs):
            job_start_vars: List[cp_model.IntVar] = []
            prev_start: cp_model.IntVar | None = None
            prev_dur = 0

            for k, (m, dur) in enumerate(job):
                # Start variable: 0 .. horizon
                st = model.NewIntVar(0, horizon, f'start_{j}_{k}')
                # Build an interval directly from start, length and computed end
                interval = model.NewIntervalVar(st, dur, st + dur, f'op_{j}_{k}')

                # Separate out the operation’s start for further use
                job_start_vars.append(st)

                # Add the precedence constraint between consecutive ops
                if prev_start is not None:
                    model.Add(st >= prev_start + prev_dur)

                # Register the interval to its machine
                machine_intervals[m].append(interval)

                # Update for next iteration
                prev_start = st
                prev_dur = dur

            starts.append(job_start_vars)

        # No‑overlap on each machine
        for intv in machine_intervals:
            model.AddNoOverlap(intv)

        # Makespan minimisation
        makespan = model.NewIntVar(0, horizon, "makespan")
        last_ops = [interval.GetEndVar() for job, intv in zip(jobs, machine_intervals) for interval in intv[-1:]]
        model.AddMaxEquality(makespan, last_ops)
        model.Minimize(makespan)

        # Solve
        solver = cp_model.CpSolver()
        status = solver.Solve(model)

        if status != cp_model.OPTIMAL:
            return []

        # Extract start times
        solution: List[List[int]] = [
            [solver.Value(st) for st in job_starts] for job_starts in starts
        ]

        return solution