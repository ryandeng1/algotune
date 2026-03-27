from scipy.linalg import solve_sylvester


class Solver:
    def solve(self, problem):
        A, B, Q = problem["A"], problem["B"], problem["Q"]
        X = solve_sylvester(A, B, Q)
        return {"X": X}