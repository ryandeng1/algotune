import cvxpy as cp
import numpy as np

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        X = np.array(problem["X"], dtype=float)
        y = np.array(problem["y"], dtype=float)
        quantile = problem["quantile"]
        fit_intercept = problem["fit_intercept"]
        
        n_samples, n_features = X.shape
        
        beta = cp.Variable(n_features)
        intercept = cp.Variable(1) if fit_intercept else None
        u = cp.Variable(n_samples)
        v = cp.Variable(n_samples)
        
        objective = cp.sum(quantile * u + (1 - quantile) * v)
        
        constraints = []
        for i in range(n_samples):
            if fit_intercept:
                constraints.append(u[i] >= y[i] - (X[i] @ beta + intercept))
                constraints.append(v[i] >= (X[i] @ beta + intercept) - y[i])
            else:
                constraints.append(u[i] >= y[i] - (X[i] @ beta))
                constraints.append(v[i] >= (X[i] @ beta) - y[i])
        
        prob = cp.Problem(cp.Minimize(objective), constraints)
        prob.solve(solver='ECOS')
        
        if fit_intercept:
            coef = beta.value
            intercept_val = intercept.value[0]
            predictions = X @ coef + intercept_val
        else:
            coef = beta.value
            predictions = X @ coef
        
        return {
            "coef": [coef.tolist()],
            "intercept": [intercept_val] if fit_intercept else [],
            "predictions": predictions.tolist()
        }
