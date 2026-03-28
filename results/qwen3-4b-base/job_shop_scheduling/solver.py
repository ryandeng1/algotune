from typing import Any
from ortools.sat.python import cp_model


class Solver:
    def solve(self, problem: dict[str, Any]) -> list[list[int]]:
        """
        Solve the JSSP using CP-SAT with interval variables and no-overlap.

        :param problem: Dict with "num_machines" and "jobs".
        :return: A list of J lists of start times for each operation.
        """
        M = problem["num_machines"]
        jobs_data = problem["jobs"]

        model = cp_model.CpModel()
        horizon = sum(d for job in jobs_data for _, d in job)

        all_tasks = {}  # (j,k) -> (start_var, duration)
        machine_to_intervals: dict[int, list[cp_model.IntervalVar]] = {m: [] for m in range(M)}
        for j, job in enumerate(jobs_data):
            for k, (m, p) in enumerate(job):
                suffix = f"_{j}_{k}"
                start = model.NewIntVar(0, horizon, f"start{suffix}")
                end = model.NewIntVar(0, horizon, f"end{suffix}")
                interval = model.NewIntervalVar(start, p, end, f"interval{suffix}")
                all_tasks[(j, k)] = (start, p)
                machine_to_intervals[m].append(interval)
                if k > 0:
                    prev_start, prev_p = all_tasks[(j, k - 1)]
                    model.Add(start >= prev_start + prev_p)

        for m in range(M):
            model.AddNoOverlap(machine_to_intervals[m])

        makespan = model.NewIntVar(0, horizon, "makespan")
        last_expressions = []
        for job_id, job in enumerate(jobs_data):
            start_last, p_last = all_tasks[(job_id, len(job) - 1)]
            last_expressions.append(start_last + p_last)
        model.AddMaxEquality(makespan, last_expressions)
        model.Minimize(makespan)

        solver = cp_model.CpSolver()
        status = solver.Solve(model)

        if status == cp_model.OPTIMAL:
            solution = []
            for j, job in enumerate(jobs_data):
                starts = [int(solver.Value(all_tasks[(j, k)][0])) for k, _ in enumerate(job)]
                solution.append(starts)
            return solution
        else:
            return []