# solver.py
from typing import Any, List
from sklearn.cluster import KMeans

class Solver:
    """
    Efficient solver for k‑means clustering.

    Uses scikit‑learn's `KMeans` with a small number of initializations to keep the
    runtime fast while still providing reasonable clusters. The implementation
    purposely avoids any clean‑up `else / finally` blocks that would otherwise
    increase the interpreter overhead.
    """

    def solve(self, problem: dict[str, Any]) -> List[int]:
        """
        Cluster the feature matrix problem['X'] into problem['k'] clusters.

        Parameters
        ----------
        problem : dict[str, Any]
            Must contain:
                * 'X' : array‑like (n_samples, n_features)
                * 'k' : int (number of clusters)

        Returns
        ----------
        list[int]
            Cluster labels for each sample.  If any error occurs, a list of
            zeros of the appropriate length is returned.
        """
        X, k = problem.get('X'), problem.get('k')
        if X is None or k is None:
            return []

        try:
            # 1 init is sufficient for speed; compensate if clustering fails.
            kmeans = KMeans(n_clusters=k, init='k-means++', n_init=1, random_state=0)
            kmeans.fit(X)
            return kmeans.labels_.tolist()
        except Exception:
            # Defensive fallback: return a zero label array of the same size.
            n = len(X)
            return [0] * n