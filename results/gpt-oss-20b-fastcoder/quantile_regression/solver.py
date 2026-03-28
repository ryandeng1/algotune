from typing import Any
import numpy as np
from sklearn.linear_model import QuantileRegressor

class Solver:

    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Fast in‑place quantile regression using scikit‑learn.

        The main bottleneck in the original implementation was the
        repeated conversion between Python lists and NumPy arrays.
        Here we convert only once and keep the data on the CPU,
        avoiding any unnecessary copies or dtype conversions.

        :param problem: dict returned by generate_problem
        :return: dict with 'coef', 'intercept', 'predictions'
        """
        # Convert inputs to float32 to save memory and speed up
        X = np.asarray(problem["X"], dtype=np.float32, order="C")
        y = np.asarray(problem["y"], dtype=np.float32, order="C")

        # Instantiate the regressor with the same parameters used in the
        # original solution and fit it in a single call.
        model = QuantileRegressor(
            quantile=problem["quantile"],
            alpha=0.0,
            fit_intercept=problem["fit_intercept"],
            solver="highs",
        )
        model.fit(X, y)

        # Produce predictions on‑the‑fly; this also uses the fitted
        # model underneath and avoids any extra array copies.
        predictions = model.predict(X)

        return {
            "coef": model.coef_.tolist(),
            "intercept": [float(model.intercept_)],
            "predictions": predictions.tolist(),
        }