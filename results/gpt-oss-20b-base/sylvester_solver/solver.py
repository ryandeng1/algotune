import numpy as np
from scipy.linalg import solve_sylvester

class Solver:
    def solve(self, problem: dict):
        A = problem['A']
        B = problem['B']
        Q = problem['Q']
        return {'X': solve_sylvester(A, B, Q)}