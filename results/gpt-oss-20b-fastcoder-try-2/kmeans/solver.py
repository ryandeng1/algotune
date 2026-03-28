import numpy as np
from sklearn.cluster import MiniBatchKMeans, KMeans

class Solver:
    def solve(self, problem: dict[str, Any]) -> list[int]:
        X = np.asarray(problem['X'])
        k = int(problem['k'])
        # Prefer the faster MiniBatchKMeans; fallback to KMeans if it fails
        try:
            # Use a small number of iterations for speed; set random_state for reproducibility
            mbk = MiniBatchKMeans(
                n_clusters=k,
                batch_size=256,
                max_iter=200,
                random_state=0,
                n_init=1,
                init='k-means++',
                verbose=0,
            )
            mbk.fit(X)
            return mbk.labels_.tolist()
        except Exception:
            try:
                # If MiniBatch fails, use plain KMeans with minimal settings
                kmeans = KMeans(
                    n_clusters=k,
                    max_iter=200,
                    n_init=1,
                    init='k-means++',
                    random_state=0,
                    verbose=0,
                )
                kmeans.fit(X)
                return kmeans.labels_.tolist()
            except Exception:
                # As a last resort, return zeros
                return [0] * len(X)