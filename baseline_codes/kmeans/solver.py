from typing import Any
import sklearn


class Solver:
    def solve(self, problem: dict[str, Any]) -> list[int]:
        try:
            # use sklearn.cluster.KMeans to solve the task
            kmeans = sklearn.cluster.KMeans(n_clusters=problem["k"]).fit(problem["X"])
            return kmeans.labels_.tolist()
        except Exception as e:
            n = len(problem["X"])
            return [0] * n  # return trivial answer
