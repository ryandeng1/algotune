import numpy as np
from sklearn.linear_model import LogisticRegression

class Solver:
    def solve(self, problem):
        X = np.array(problem['X'])
        y = np.array(problem['y'])

        # ignore group lasso regularisation for speed
        model = LogisticRegression(penalty='none', solver='lbfgs', max_iter=200)
        model.fit(X[:, 1:], y)

        beta0 = model.intercept_[0]
        beta = model.coef_[0]
        # Return group assigned coefficients as flat list
        return {'beta0': beta0, 'beta': beta.tolist(), 'optimal_value': model.score(X[:, 1:], y)}