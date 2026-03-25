# solver.py
from typing import Any, Dict, List, Tuple
from ortools.sat.python import cp_model


class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> List[List[int]]:
        """
        Solve a Job Shop Scheduling Problem (JSSP) using OR-Tools CP-SAT.

        Parameters
        ----------
        problem : dict
            Dictionary with keys:
                "num_machines" : int, number of machines (0-indexed).
                "jobs" : List[List[Tuple[int, int]]]
                    Each entry is a list of (machine, duration) pairs for one job.

        Returns
        -------
        List[List[int]]
            List of J lists where each inner list contains the start times for
            the operations of that job.  The lengths match the job operation
            counts and the schedule is optimal.
        """
        # Extract problem data
        M = problem["num_machines"]
        jobs_data = problem["jobs"]

        model = cp_model.CpModel()

        # Rough upper bound for the horizon
        horizon = sum(d for job in jobs_data for _, d in job)

        # Keep track of all tasks and intervals per machine
        all_tasks: Dict[Tuple[int, int], Tuple[cp_model.IntVar, cp_model.IntVar, int]] = {}
        machine_to_intervals: Dict[int, List[cp_model.IntervalVar]] = {m: [] for m in range(M)}

        for j, job in enumerate(jobs_data):
            prev_end = None
            for k, (m, p) in enumerate(job):
                suffix = f"_{j}_{k}"
                start = model.NewIntVar(0, horizon, f"start{suffix}")
                end = model.NewIntVar(0, horizon, f"end{suffix}")
                interval = model.NewIntervalVar(start, p, end, f"interval{suffix}")

                all_tasks[(j, k)] = (start, end, p)
                machine_to_intervals[m].append(interval)

                if prev_end is not None:
                    model.Add(start >= prev_end)
                prev_end = end

        # No-overlap constraints per machine
        for m in range(M):
            model.AddNoOverlap(machine_to_intervals[m])

        # Makespan definition
        makespan = model.NewIntVar(0, horizon, "makespan")
        last_ends = [all_tasks[(j, len(job) - 1)][1] for j, job in enumerate(jobs_data)]
        model.AddMaxEquality(makespan, last_ends)
        model.Minimize(makespan)

        # Solve
        solver = cp_model.CpSolver()
        status = solver.Solve(model)

        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            solution = []
            for j, job in enumerate(jobs_data):
                starts = [int(solver.Value(all_tasks[(j, k)][0])) for k in range(len(job))]
                solution.append(starts)
            return solution
        else:
            return []
