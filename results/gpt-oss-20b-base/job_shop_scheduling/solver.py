# solver.py

from typing import Any, List, Dict
from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> List[List[int]]:
        """
        Solve Job Shop Scheduling Problem (JSSP) optimally using OR-Tools CP-SAT.

        :param problem: Dictionary with keys:
                        "num_machines": int
                        "jobs": List[List[Tuple[int, int]]]
        :return: List of start times per job, matching CP-SAT solution.
        """
        M = problem["num_machines"]
        jobs_data = problem["jobs"]

        model = cp_model.CpModel()

        # Compute horizon: sum of all durations
        horizon = sum(d for job in jobs_data for _, d in job)

        # Store interval variables and machine assignments
        all_tasks: Dict[tuple, tuple] = {}  # (job, op) -> (start, end, duration)
        machine_intervals: Dict[int, List[cp_model.IntervalVar]] = {m: [] for m in range(M)}

        for j, job in enumerate(jobs_data):
            for k, (m, p) in enumerate(job):
                suffix = f"_{j}_{k}"
                start = model.NewIntVar(0, horizon, f"start{suffix}")
                end = model.NewIntVar(0, horizon, f"end{suffix}")
                interval = model.NewIntervalVar(start, p, end, f"interval{suffix}")

                all_tasks[(j, k)] = (start, end, p)
                machine_intervals[m].append(interval)

                # Precedence constraint within job
                if k > 0:
                    prev_end = all_tasks[(j, k - 1)][1]
                    model.Add(start >= prev_end)

        # No-overlap constraints per machine
        for m in range(M):
            model.AddNoOverlap(machine_intervals[m])

        # Makespan variable and objective
        makespan = model.NewIntVar(0, horizon, "makespan")
        last_ends = [all_tasks[(j, len(jobs_data[j]) - 1)][1] for j in range(len(jobs_data))]
        model.AddMaxEquality(makespan, last_ends)
        model.Minimize(makespan)

        # Solve
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = kwargs.get("timeout", 300)  # optional timeout
        status = solver.Solve(model)

        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            solution: List[List[int]] = []
            for j, job in enumerate(jobs_data):
                starts = [
                    int(solver.Value(all_tasks[(j, k)][0]))
                    for k, _ in enumerate(job)
                ]
                solution.append(starts)
            return solution
        else:
            return []
