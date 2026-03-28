from typing import Any
import cvxpy as cp
import numpy as np

class Solver:

    def solve(self, problem: dict[str, list[list[int]] | list[float] | int]) -> dict[str, list[list[float]] | float]:
        """
        Solves the Perron-Frobenius matrix completion using CVXPY.

        Args:
            problem: Dict containing inds, a.

        Returns:
            Dict with estimates B, optimal_value.
        """
        inds = np.array(problem['inds'])
        a = np.array(problem['a'])
        n = problem['n']
        xx, yy = np.meshgrid(np.arange(n), np.arange(n))
        allinds = np.vstack((yy.flatten(), xx.flatten())).T
        otherinds = allinds[~(allinds == inds[:, None]).all(2).any(0)]
        B = cp.Variable((n, n), pos=True)
        objective = cp.Minimize(cp.pf_eigenvalue(B))
        constraints = [cp.prod(B[otherinds[:, 0], otherinds[:, 1]]) == 1.0, B[inds[:, 0], inds[:, 1]] == a]
        prob = cp.Problem(objective, constraints)
        try:
            result = prob.solve(gp=True)
        except cp.SolverError as e:
            return None
        except Exception as e:
            return None
        else:
            pass
        finally:
            pass
        if B.value is None:
            return None
        else:
            pass
        return {'B': B.value.tolist(), 'optimal_value': result}
