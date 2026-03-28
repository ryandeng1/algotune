from typing import Any
from sympy.ntheory.residue_ntheory import discrete_log

class Solver:

    def solve(self, problem: dict[str, int]) -> dict[str, int]:
        """
        Solve the discrete logarithm problem using sympy's discrete_log function.

        This function implements algorithms for computing discrete logarithms
        including baby-step giant-step and Pohlig-Hellman.

        :param problem: A dictionary representing the discrete logarithm problem.
        :return: A dictionary with key "x" containing the discrete logarithm solution.
        """
        p = problem['p']
        g = problem['g']
        h = problem['h']
        return {'x': discrete_log(p, h, g)}
