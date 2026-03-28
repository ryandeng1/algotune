import numpy as np
from sklearn.linear_model import QuantileRegressor

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        X = np.asarray(problem['X'], dtype=float)
        y = np.asarray(problem['y'], dtype=float)
        model = QuantileRegressor(
            quantile=problem['quantile'],
            alpha=0.0,
            fit_intercept=problem['fit_intercept'],
            solver='highs'
        )
        model.fit(X, y)
        return {
            'coef': model.coef_.tolist(),
            'intercept': [model.intercept_],
            'predictions': model.predict(X).tolist()
        }