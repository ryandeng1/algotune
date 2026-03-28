from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem: dict) -> list[list[int]]:
        """Solve JSSP with CP-SAT, returning start times per job."""
        M = problem["num_machines"]
        jobs = problem["jobs"]
        n_jobs = len(jobs)

        # Compute horizon once
        horizon = sum(p for job in jobs for _, p in job)

        model = cp_model.CpModel()

        # Prepare data structures
        starts = [[None] * len(job) for job in jobs]
        ends = [[None] * len(job) for job in jobs]
        machine_intervals = [[] for _ in range(M)]

        for j, job in enumerate(jobs):
            prev_end = None
            for k, (m, p) in enumerate(job):
                s = model.NewIntVar(0, horizon, f"s_{j}_{k}")
                e = model.NewIntVar(0, horizon, f"e_{j}_{k}")
                interval = model.NewIntervalVar(s, p, e, f"int_{j}_{k}")
                starts[j][k] = s
                ends[j][k] = e
                machine_intervals[m].append(interval)
                if prev_end is not None:
                    model.Add(s >= prev_end)
                prev_end = e

        # No overlapping on each machine
        for m in range(M):
            model.AddNoOverlap(machine_intervals[m])

        # Makespan minimization
        makespan = model.NewIntVar(0, horizon, "makespan")
        final_end = [ends[j][len(job) - 1] for j, job in enumerate(jobs)]
        model.AddMaxEquality(makespan, final_end)
        model.Minimize(makespan)

        solver = cp_model.CpSolver()
        status = solver.Solve(model)

        if status != cp_model.OPTIMAL:
            return []

        # Build solution
        solution = [[int(solver.Value(starts[j][k])) for k in range(len(jobs[j]))] for j in range(n_jobs)]
        return solution