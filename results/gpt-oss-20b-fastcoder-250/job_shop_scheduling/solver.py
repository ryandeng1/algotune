from typing import Any, List
from ortools.sat.python import cp_model

def solve(problem: dict[str, Any]) -> List[List[int]]:
    """
    Optimised CP-SAT solver for a JSSP instance.

    The implementation replaces numerous dictionary look‑ups with
    flat python lists indexed by job and operation numbers, reducing
    Python overhead and speeding up the construction of the model.
    """
    M = problem["num_machines"]
    jobs = problem["jobs"]

    model = cp_model.CpModel()

    # Compute an upper bound for all start times (the horizon)
    horizon = sum(d for job in jobs for _, d in job)

    # ------------------------------------------------------------------
    # Data structures
    # ------------------------------------------------------------------
    # start[j][k]         -> start variable of operation k in job j
    start: List[List[cp_model.IntVar]] = [[] for _ in range(len(jobs))]
    # operation duration[j][k]  -> fixed duration
    duration: List[List[int]] = [[] for _ in range(len(jobs))]
    # machine intervals per machine
    machine_intervals: List[List[cp_model.IntervalVar]] = [[] for _ in range(M)]

    # ------------------------------------------------------------------
    # Build variables, intervals and precedence constraints
    # ------------------------------------------------------------------
    for j, job in enumerate(jobs):
        prev_end: cp_model.IntVar | None = None
        for k, (m, p) in enumerate(job):
            # Create variables
            s = model.NewIntVar(0, horizon, f"s_{j}_{k}")
            e = model.NewIntVar(0, horizon, f"e_{j}_{k}")
            iv = model.NewIntervalVar(s, p, e, f"iv_{j}_{k}")

            # Store
            start[j].append(s)
            duration[j].append(p)
            machine_intervals[m].append(iv)

            # Precedence constraint inside the job
            if prev_end is not None:
                model.Add(s >= prev_end)
            prev_end = e

    # No-overlap constraints per machine
    for intrv in machine_intervals:
        model.AddNoOverlap(intrv)

    # ------------------------------------------------------------------
    # Makespan objective
    # ------------------------------------------------------------------
    makespan = model.NewIntVar(0, horizon, "makespan")
    last_ends = [start[j][-1] + duration[j][-1] for j in range(len(jobs))]
    model.AddMaxEquality(makespan, last_ends)
    model.Minimize(makespan)

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        return [[int(solver.Value(s)) for s in ops] for ops in start]
    return []