from typing import Any
import faiss
import numpy as np

# The solver is intentionally lightweight – it simply wraps faiss's k‑means
# implementation and returns the requested information in the format expected
# by the benchmark harness.  The code is written so that it does the least
# work necessary: the input is converted to a contiguous float32 array only
# once, the k‑means is run with a moderate number of iterations, and the
# result structures are built only after the expensive calculations are
# finished.

class Solver:

    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Perform vector quantization using faiss Kmeans and return centroids,
        assignments, and the mean squared quantization error.

        :param problem: dictionary containing:
            - 'vectors': list of vectors (list[list[float]] or numpy array)
            - 'k': number of centroids
        :return: dictionary with keys:
            - 'centroids': list of centroids as lists of floats
            - 'assignments': list of integer indices (one per input vector)
            - 'quantization_error': mean squared L2 error
        """
        vectors = np.asarray(problem['vectors'], dtype=np.float32, order='C')
        k = int(problem['k'])
        dim = vectors.shape[1]

        # faiss.Kmeans: the default number of iterations (10) is usually
        # sufficient for small problems but to guarantee a good fit we use
        # a slightly larger value that still keeps training fast.
        kmeans = faiss.Kmeans(dim, k, niter=20, verbose=False, nredo=1)
        kmeans.train(vectors)

        centroids = kmeans.centroids  # dtype will already be float32
        index = faiss.IndexFlatL2(dim)
        index.add(centroids)

        # distances has shape (len(vectors), 1) – we only need the first axis.
        distances, assignments = index.search(vectors, 1)
        quantization_error = float(np.mean(distances))

        return {
            'centroids': centroids.tolist(),
            'assignments': assignments.flatten().tolist(),
            'quantization_error': quantization_error,
        }