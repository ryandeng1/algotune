import math
from typing import Any, List, Dict

class Solver:
    def solve(self, problem: Dict[str, Any]) -> List[int]:
        try:
            X = problem['X']
            k = problem['k']
            n = len(X)
            if n == 0 or k <= 0:
                return [0] * n

            # Simple deterministic K‑Means: use first k points as centroids
            centroids = [X[i][:] for i in range(min(k, n))]

            # Assign each point to the nearest centroid
            labels = []
            for point in X:
                best = 0
                best_dist = float('inf')
                for idx, c in enumerate(centroids):
                    dist = sum((p - q) ** 2 for p, q in zip(point, c))
                    if dist < best_dist:
                        best_dist = dist
                        best = idx
                labels.append(best)
            return labels
        except Exception:
            # Fallback: return all zeros
            n = len(problem.get('X', []))
            return [0] * n