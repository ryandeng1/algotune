# solver.py
from __future__ import annotations
from typing import Any, List, Dict, Tuple

from ortools.sat.python import cp_model


class Solver:
    """
    A very small CP‑SAT based JSSP solver.
    """

    def solve(self, problem: Dict[str, Any]) -> List[List[int]]:
        """
        Solve the Job‑Shop Scheduling Problem (JSSP) using OR‑Tools CP‑SAT.
        :param problem: {'num_machines': int, 'jobs': List[List[Tuple[int, int]]]}
        :return: 2‑D list.  Outer list index=job, inner list: start times of each operation.
        """
        M = problem["num_machines"]
        jobs = problem["jobs"]

        # Summed horizon: maximum possible makespan
        horizon = sum(p for job in jobs for _, p in job)

        model = cp_model.CpModel()

        # Create interval vars and store starts in a compact dict
        starts: Dict[Tuple[int, int], int] = {}
        ends: Dict[Tuple[int, int], int] = {}
        machine_intervals: List[List[cp_model.IntervalVar]] = [[] for _ in range(M)]

        # Prepare all operations
        for job_id, job in enumerate(jobs):
            prev_end = None
            for op_id, (machine, duration) in enumerate(job):
                suffix = f"_{job_id}_{op_id}"
                start_var = model.NewIntVar(0, horizon, f"start{suffix}")
                end_var = model.NewIntVar(0, horizon, f"end{suffix}")
                interval = model.NewIntervalVar(start_var, duration, end_var, f"interval{suffix}")

                starts[(job_id, op_id)] = start_var
                ends[(job_id, op_id)] = end_var

                machine_intervals[machine].append(interval)

                if prev_end is not None:
                    # enforce operation order on the same job
                    model.Add(start_var >= prev_end)
                prev_end = end_var

        # No‑overlap constraints per machine
        for machine_ix in range(M):
            model.AddNoOverlap(machine_intervals[machine_ix])

        # Makespan minimisation
        makespan = model.NewIntVar(0, horizon, "makespan")
        last_ops = [ends[(job_id, len(job) - 1)] for job_id, job in enumerate(jobs)]
        model.AddMaxEquality(makespan, last_ops)
        model.Minimize(makespan)

        # Solve
        solver = cp_model.CpSolver()
        status = solver.Solve(model)

        if status == cp_model.OPTIMAL:
            # Build solution list
            solution: List[List[int]] = [
                [int(solver.Value(starts[(i, j)])) for j in range(len(job))]
                for i, job in enumerate(jobs)
            ]
            return solution
        return []