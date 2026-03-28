from typing import Any
import numpy as np
from sklearn.linear_model import QuantileRegressor

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Fit a quantile regression model using scikit‑learn's
        QuantileRegressor (solver high‑s based linear programming),
        then return model parameters and predictions.

        Parameters
        ----------
        problem : dict
            Must contain
            - 'X'   : matrix of predictors
            - 'y'   : target vector
            - 'quantile' : target quantile (0 < q < 1)
            - 'fit_intercept' : bool, whether to fit intercept

        Returns
        -------
        dict
            {'coef' : list, 'intercept' : list, 'predictions' : list}
        """
        X = np.asarray(problem['X'], dtype=np.float64)
        y = np.asarray(problem['y'], dtype=np.float64)

        # Fit model
        model = QuantileRegressor(
            quantile=problem['quantile'],
            alpha=0.0,
            fit_intercept=problem['fit_intercept'],
            solver='highs'
        )
        model.fit(X, y)

        # Convert outputs to list for consistency
        return {
            'coef': model.coef_.tolist(),
            'intercept': [model.intercept_],
            'predictions': model.predict(X).tolist()
        }