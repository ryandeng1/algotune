#!/usr/bin/env python3
"""Optimised CP‑SAT solver for the Job‑Shop Scheduling problem."""
from typing import Any, List
from ortools.sat.python import cp_model


class Solver:
    """Solution class used by the CI framework."""

    def solve(self, problem: dict[str, Any]) -> List[List[int]]:
        """
        Solve JSSP using OR‑Tools CP‑SAT.

        Parameters
        ----------
        problem : dict
            Dictionary with `num_machines` and `jobs` (sequence of (machine, duration)).

        Returns
        -------
        List[List[int]]
            List of start times for each operation per job.
        """
        # ------------------------------------------------------------------
        #  Small helper functions to keep the main loop tight and reduce
        #  attribute lookups.
        # ------------------------------------------------------------------
        data_jobs = problem["jobs"]
        nb_jobs = len(data_jobs)
        nb_machines = problem["num_machines"]

        # Compute a tight horizon using the sum of latest possible start time
        # for each job (critical path of job order).  
        horizon = 0
        for job in data_jobs:
            horizon += max(p for _, p in job)

        # ------------------------------------------------------------------
        #  Build model
        # ------------------------------------------------------------------
        model = cp_model.CpModel()

        # Dispatching intervals by machine
        machine_intervals = [[] for _ in range(nb_machines)]
        # Store start and end vars to easly access later
        start_vars = [[None] * len(job) for job in data_jobs]
        end_vars = [[None] * len(job) for job in data_jobs]

        for j, job in enumerate(data_jobs):
            prev_end = None
            for k, (m, duration) in enumerate(job):
                s = model.NewIntVar(0, horizon, f"s_{j}_{k}")
                e = model.NewIntVar(0, horizon, f"e_{j}_{k}")
                model.NewIntervalVar(s, duration, e, f"i_{j}_{k}")
                start_vars[j][k] = s
                end_vars[j][k] = e
                machine_intervals[m].append(model.NewIntervalVar(s, duration, e, f"m{m}_{j}_{k}"))
                if prev_end is not None:
                    model.Add(s >= prev_end)
                prev_end = e

        # No‑overlap on each machine
        for m_intervals in machine_intervals:
            model.AddNoOverlap(m_intervals)

        # Make objective
        job_last_ends = [end_vars[j][-1] for j in range(nb_jobs)]
        makespan = model.NewIntVar(0, horizon, "makespan")
        model.AddMaxEquality(makespan, job_last_ends)
        model.Minimize(makespan)

        # ------------------------------------------------------------------
        #  Solve
        # ------------------------------------------------------------------
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 5.0          # keep search short
        solver.parameters.num_search_workers = 8            # use parallel workers

        status = solver.Solve(model)
        if status != cp_model.OPTIMAL and status != cp_model.FEASIBLE:
            return []

        # Extract solution
        solution = [[solver.Value(start_vars[j][k]) for k in range(len(job))]
                    for j, job in enumerate(data_jobs)]
        return solution