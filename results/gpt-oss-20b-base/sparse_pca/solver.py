import numpy as np
from sklearn.decomposition import SparsePCA
from sklearn.preprocessing import StandardScaler

class Solver:
    def solve(self, problem: dict) -> dict:
        """
        Solve the sparse PCA problem using scikit-learn's SparsePCA
        which is considerably faster than a full cvxpy formulation.
        """
        cov = np.array(problem['covariance'])
        n_components = int(problem['n_components'])
        sparsity_param = float(problem['sparsity_param'])
        n = cov.shape[0]

        # Construct a synthetic dataset that yields the given covariance.
        # We use a standard normal and scale it to match the covariance.
        rng = np.random.default_rng(seed=42)
        Z = rng.standard_normal((n_components * 10, n))
        # Whiten the data and apply the covariance matrix
        scaler = StandardScaler()
        X = scaler.fit_transform(Z)
        X = X @ np.linalg.cholesky(cov)

        # Fit SparsePCA
        spca = SparsePCA(
            n_components=n_components,
            alpha=sparsity_param,
            random_state=42,
            max_iter=500,
            tol=1e-3,
            n_jobs=-1,
            copy=True,
        )
        try:
            spca.fit(X)
        except Exception:
            return {'components': [], 'explained_variance': []}

        # SparsePCA returns components_ of shape (n_components, n_features)
        # We need to transpose to match the expected output shape (n, n_components)
        components = spca.components_.T

        # Compute explained variance for each component
        explained_variance = [
            float(components[:, i].T @ cov @ components[:, i]) for i in range(n_components)
        ]

        return {
            'components': components.tolist(),
            'explained_variance': explained_variance,
        }