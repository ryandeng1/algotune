# solver.py
from __future__ import annotations
from typing import Any, Dict, List
import faiss
import numpy as np

# A tiny helper to avoid re‑allocating FP constants
_FLOAT32_ZERO = np.float32(0.0)


class Solver:
    """
    Fast vector quantization using Faiss k‑means.
    """

    def __init__(self) -> None:
        # No persistent state needed – everything is created in solve().
        pass

    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run k‑means clustering on the supplied vectors and return a compact result.

        Parameters
        ----------
        problem : dict
            Must contain:
            - 'vectors': list[list[float]] – 2‑D matrix of input points
            - 'k': int – desired number of centroids

        Returns
        -------
        dict
            {
                "centroids": List[List[float]]
                "assignments": List[int]
                "quantization_error": float
            }
        """
        # Fast Numpy conversion – avoid temporary Python objects
        vectors: np.ndarray = np.array(problem["vectors"], dtype=np.float32, copy=False)
        k: int = int(problem["k"])

        # Faiss k‑means (fewer iterations is usually enough for small problems)
        dim = vectors.shape[1]
        kmeans = faiss.Kmeans(
            dim,
            k,
            niter=20,          # fewer iterations → much faster
            verbose=False,
            seed=0,            # deterministic
        )
        kmeans.train(vectors)

        centroids: np.ndarray = kmeans.centroids  # shape (k, dim)

        # Nearest‑centroid search – only one neighbor needed
        index = faiss.IndexFlatL2(dim)
        index.add(centroids)                       # add centroids once

        distances, assignments = index.search(vectors, 1)  # assignments shape (n,1)

        # Mean squared error (distances already L2–squared)
        mse = np.mean(distances, dtype=np.float64) if distances.size else _FLOAT32_ZERO

        # Convert to Python lists – minimal overhead
        return {
            "centroids": centroids.tolist(),
            "assignments": assignments.flatten().tolist(),
            "quantization_error": float(mse),
        }