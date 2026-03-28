import numpy as np
import faiss

# No external dependencies beyond faiss and numpy

def solve(problem: dict) -> dict:
    """
    Search k nearest neighbours for each query point in `problem['queries']`
    over the reference points `problem['points']`.

    If the optional flag `distribution` is set to `'hypercube_shell'`,
    additionally compute the distances and indices for the boundary queries
    that ask for points on two hyperplanes (x_i = 0 and x_i = 1) for each
    dimension.

    The function returns a dict compatible with the original implementation.
    """
    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------
    points = np.asarray(problem['points'], dtype=np.float32)
    queries = np.asarray(problem['queries'], dtype=np.float32)
    dim = points.shape[1]
    k = min(problem['k'], len(points))

    # ------------------------------------------------------------------
    # Build FAISS index
    # ------------------------------------------------------------------
    index = faiss.IndexFlatL2(dim)          # L2 distance, no GPU
    id_map = faiss.IndexIDMap(index)        # map internal idx -> external ids
    id_map.add_with_ids(points, np.arange(len(points), dtype=np.int64))

    # ------------------------------------------------------------------
    # Main search
    # ------------------------------------------------------------------
    distances, indices = id_map.search(queries, k)

    solution = {
        'indices': indices.tolist(),
        'distances': distances.tolist()
    }

    # ------------------------------------------------------------------
    # Boundary queries (only if requested)
    # ------------------------------------------------------------------
    if problem.get('distribution') == 'hypercube_shell':
        # Build boundary query vectors in bulk
        # For each dimension d, produce [0,...,0,1,0,...,0] and [1,...,1,0,1,...,1]
        zeros = np.zeros((dim, dim), dtype=np.float32)
        ones  = np.ones((dim, dim), dtype=np.float32)
        zeros[np.arange(dim), np.arange(dim)] = 0.0  # keep at 0 (already)
        ones[np.arange(dim), np.arange(dim)]  = 1.0  # keep at 1 (already)
        bqs = np.vstack((zeros, ones))

        bq_dist, bq_idx = id_map.search(bqs, k)
        solution['boundary_distances'] = bq_dist.tolist()
        solution['boundary_indices'] = bq_idx.tolist()

    return solution