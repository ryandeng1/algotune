from typing import Any, Dict, List
from ortools.sat.python import cp_model


class Solver:
    """
    CP‑SAT solver for the job shop scheduling problem.
    Works for small to medium instances (≤ 50 jobs, ≤ 20 machines).
    """
    def solve(self, problem: Dict[str, Any]) -> List[List[int]]:
        """
        Return the earliest start time for each operation in the optimal schedule.

        Parameters
        ----------
        problem : dict
            Dictionary with keys:
                - "num_machines" : int
                - "jobs" : list of list of tuples (machine_index, duration)

        Returns
        -------
        list[list[int]] or []
            Nested list of job start times. Empty list if the model is infeasible.
        """

        num_machines, jobs = problem["num_machines"], problem["jobs"]

        # Build CP‑SAT model
        model = cp_model.CpModel()
        # Upper bound for horizon: sum of all durations
        horizon = sum(d for job in jobs for _, d in job)

        # Create tasks and gather machine intervals
        machine_intervals: Dict[int, List[cp_model.IntervalVar]] = {m: [] for m in range(num_machines)}
        task_start = {}
        for j, job in enumerate(jobs):
            for k, (m, p) in enumerate(job):
                # Interval variable
                start = model.NewIntVar(0, horizon, f'start_{j}_{k}')
                end = model.NewIntVar(0, horizon, f'end_{j}_{k}')
                interval = model.NewIntervalVar(start, p, end, f'interval_{j}_{k}')

                task_start[(j, k)] = start
                machine_intervals[m].append(interval)

                # Precedence constraints within the same job
                if k > 0:
                    prev_end = task_start[(j, k - 1)].Index() + jobs[j][k - 1][1]
                    model.Add(start >= prev_end)

        # No‑overlap on each machine
        for m in range(num_machines):
            model.AddNoOverlap(machine_intervals[m])

        # Makespan computation
        makespan = model.NewIntVar(0, horizon, 'makespan')
        last_ends = [task_start[(j, len(job) - 1)].Index() + job[-1][1]
                     for j, job in enumerate(jobs)]
        model.AddMaxEquality(makespan, last_ends)

        # Objective: minimize makespan
        model.Minimize(makespan)

        # Solve
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 30.0  # optional time limit

        status = solver.Solve(model)
        if status != cp_model.OPTIMAL and status != cp_model.FEASIBLE:
            return []

        # Extract solution
        solution: List[List[int]] = []
        for j, job in enumerate(jobs):
            starts = [solver.Value(task_start[(j, k)]) for k in range(len(job))]
            solution.append(starts)

        return solution