from typing import Any, List
from ortools.sat.python import cp_model


class Solver:
    def solve(self, problem: dict[str, Any]) -> List[List[int]]:
        """
        Fast CP‑SAT solver for Job Shop Scheduling Problem (JSSP).
        Returns start times for every operation of every job.
        """
        # Extract problem data
        num_mach = problem["num_machines"]
        jobs = problem["jobs"]

        # Create model
        model = cp_model.CpModel()

        # Compute a safe horizon: sum of all durations
        horizon = sum(p for job in jobs for _, p in job)

        # Store intervals by machine and all start/end vars
        machine_intervals: dict[int, List[cp_model.IntervalVar]] = {m: [] for m in range(num_mach)}
        starts: dict[tuple[int, int], cp_model.IntVar] = {}
        ends: dict[tuple[int, int], cp_model.IntVar] = {}

        # Create variables
        for j, job in enumerate(jobs):
            prev_end = None
            for k, (m, dur) in enumerate(job):
                suffix = f"_{j}_{k}"
                start = model.NewIntVar(0, horizon, f"start{suffix}")
                end = model.NewIntVar(0, horizon, f"end{suffix}")
                model.Add(end == start + dur)

                interval = model.NewIntervalVar(start, dur, end, f"interval{suffix}")
                machine_intervals[m].append(interval)

                starts[j, k] = start
                ends[j, k] = end

                if prev_end is not None:
                    model.Add(start >= prev_end)
                prev_end = end

        # No‑overlap constraints per machine
        for intervals in machine_intervals.values():
            model.AddNoOverlap(intervals)

        # Makespan minimisation
        makespan = model.NewIntVar(0, horizon, "makespan")
        last_ops = [ends[j, len(job) - 1] for j, job in enumerate(jobs)]
        model.AddMaxEquality(makespan, last_ops)
        model.Minimize(makespan)

        # Solve
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 60.0  # optional time limit
        status = solver.Solve(model)

        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            result: List[List[int]] = []
            for j, job in enumerate(jobs):
                result.append([solver.Value(starts[j, k]) for k in range(len(job))])
            return result
        return []