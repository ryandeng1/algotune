from ortools.sat.python import cp_model

class Solver:
    # The solver works by building a CP‑SAT model that
    #   * chooses at most one B‑vertex for every A‑vertex,
    * chooses at most one A‑vertex for every B‑vertex,
    * never selects two pairs that are incompatible
    #   (i.e. have different edge values).
    #
    # The original implementation contained many "else: pass" statements
    # that were either dead code or produced an excessive number of nested
    # blocks. Those lines have been removed for clarity and speed.
    # We also replace the double summation in the binary constraints
    # with a single call to `AddAtMostOne`, which is much faster in
    # the CP‑SAT engine.

    def solve(self, problem: dict[str, list[list[int]]]) -> list[tuple[int, int]]:
        A = problem['A']
        B = problem['B']
        n, m = len(A), len(B)

        model = cp_model.CpModel()

        # binary variables x[i][p] == 1 if A[i] is matched to B[p]
        x = [[model.NewBoolVar(f'x_{i}_{p}') for p in range(m)] for i in range(n)]

        # Each A matches at most one B
        for i in range(n):
            model.AddAtMostOne(x[i])

        # Each B matches at most one A
        for p in range(m):
            model.AddAtMostOne([x[i][p] for i in range(n)])

        # Incompatible pairs: if A[i][j] != B[p][q] then not both x[i][p] and x[j][q] can be 1
        for i in range(n):
            for j in range(i + 1, n):
                # Pre‑compute the indices in B that have an incompatible value with the edge A[i][j]
                incompatible_q = [p for p in range(m) if A[i][j] != B[p][q]]
                for q in range(m):
                    # For each q, find p such that B[p][q] is incompatible
                    for p in range(m):
                        if p == q:
                            continue
                        if A[i][j] != B[p][q]:
                            model.Add(x[i][p] + x[j][q] <= 1)

        # Objective: maximize the number of matched pairs
        model.Maximize(sum(x[i][p] for i in range(n) for p in range(m)))

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 30.0  # sensible limit for small instances
        status = solver.Solve(model)

        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            # Extract solution
            return [
                (i, p)
                for i in range(n)
                for p in range(m)
                if solver.Value(x[i][p]) == 1
            ]
        return []