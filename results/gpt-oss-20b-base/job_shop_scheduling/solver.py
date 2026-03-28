from typing import Any, List, Dict
from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem: dict[str, Any]) -> List[List[int]]:
        """Solve JSSP using CP‑SAT (intervals + no‑overlap)."""
        M = problem["num_machines"]
        jobs = problem["jobs"]

        # Upper bound on makespan: sum of all durations
        horizon = sum(p for job in jobs for _, p in job)

        model = cp_model.CpModel()
        # 2‑D list of start variables: jobs x ops
        starts: List[List[cp_model.IntVar]] = []

        # Intervals scheduled on each machine
        machine_intervals: Dict[int, List[cp_model.IntervalVar]] = {m: [] for m in range(M)}

        for j, job in enumerate(jobs):
            job_starts = []
            for k, (m, p) in enumerate(job):
                suffix = f"_{j}_{k}"
                s = model.NewIntVar(0, horizon, f"start{suffix}")
                # Interval doesn't need an explicit end var
                interval = model.NewIntervalVar(s, p, s + p, f"interval{suffix}")
                machine_intervals[m].append(interval)
                job_starts.append(s)
                # Precedence constraint with previous operation
                if k > 0:
                    prev_p = job[k - 1][1]
                    model.Add(s >= job_starts[k - 1] + prev_p)
            starts.append(job_starts)

        # No‑overlap on each machine
        for m in range(M):
            model.AddNoOverlap(machine_intervals[m])

        # Makespan = max of finish times of last ops of all jobs
        makespan = model.NewIntVar(0, horizon, "makespan")
        last_end_exprs = []
        for j, job in enumerate(jobs):
            last_op = job[-1]
            last_p = last_op[1]
            last_end_exprs.append(starts[j][-1] + last_p)
        model.AddMaxEquality(makespan, last_end_exprs)
        model.Minimize(makespan)

        # Solve
        solver = cp_model.CpSolver()
        status = solver.Solve(model)

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            result: List[List[int]] = []
            for j, job_starts in enumerate(starts):
                result.append([solver.Value(v) for v in job_starts])
            return result
        return []