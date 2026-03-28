import numpy as np
from scipy.linalg import solve_sylvester

class Solver:
    def solve(self, problem):
        A, B, Q = problem['A'], problem['B'], problem['Q']
        return {'X': solve_sylvester(A, B, Q)}