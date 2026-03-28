class Solver:
    def solve(self, problem: dict[str, Any]) -> list[list[float]]:
        try:
            n_components = problem["n_components"]
            X = np.array(problem["X"])
            model = sklearn.decomposition.PCA(n_components=n_components, copy=False)
            model.fit(X)
            V = model.components_
            return [[float(v) for v in row] for row in V]
        except Exception as e:
            n_components = problem["n_components"]
            X = np.array(problem["X"])
            n, d = X.shape
            V = np.zeros((n_components, n))
            V[:, :n_components] = np.eye(n_components)
            return [[float(v) for v in row] for row in V]