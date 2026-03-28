from typing import Any
import numpy as np
from sklearn import linear_model

class Solver:

    def solve(self, problem: dict[str, Any]) -> list[float]:
        try:
            clf = linear_model.Lasso(alpha=0.1, fit_intercept=False)
            clf.fit(problem['X'], problem['y'])
            return clf.coef_.tolist()
        except Exception as e:
            _, d = problem['X'].shape
            return np.zeros(d).tolist()
        else:
            pass
        finally:
            pass
