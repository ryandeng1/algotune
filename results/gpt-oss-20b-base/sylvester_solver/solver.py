from scipy.linalg import solve_sylvester

class Solver:
    def solve(self, problem):
        A = problem["A"]
        B = problem["B"]
        Q = problem["Q"]
        return {"X": solve_sylvester(A, B, Q)}