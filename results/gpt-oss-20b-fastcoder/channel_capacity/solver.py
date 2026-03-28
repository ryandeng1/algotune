import math
import numpy as np
import cvxpy as cp
from scipy.special import xlogy

class Solver:
    def solve(self, problem: dict) -> dict:
        P = np.array(problem["P"])
        m, n = P.shape
        # P must be (n, m) – channel transition matrix (outcomes × inputs)
        if P.shape != (n, m) or not (n > 0 and m > 0):
            return None

        # Decision variable: input distribution over n symbols
        x = cp.Variable(shape=n, nonneg=True)
        y = P @ x  # distribution of outputs

        # Expected self‑information of the channel conditioned on input
        c = np.sum(xlogy(P, P), axis=0) / math.log(2.0)
        mutual_information = c @ x + cp.sum(cp.entr(y) / math.log(2.0))
        objective = cp.Maximize(mutual_information)
        constraints = [cp.sum(x) == 1]
        prob = cp.Problem(objective, constraints)

        try:
            # The HIGHS solver is fast and free; defaults to dense matrices
            prob.solve(solver=cp.GUROBI, verbose=False)
        except Exception:
            try:
                prob.solve(solver=cp.GLPK, verbose=False)
            except Exception:
                try:
                    prob.solve(solver=cp.COIN, verbose=False)
                except Exception:
                    try:
                        prob.solve(solver=cp.ECOS, verbose=False)
                    except Exception:
                        return None

        if prob.value is None:
            return None
        return {"x": x.value.tolist(), "C": prob.value}