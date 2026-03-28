import numpy as np
import scipy.sparse
import scipy.sparse.csgraph

def solve(problem: dict):
    """
    Computes the graph Laplacian using scipy.sparse.csgraph.laplacian.
    The result is returned as CSR components.
    """
    # Build CSR matrix from the input data
    try:
        data   = np.asarray(problem['data'])
        indices= np.asarray(problem['indices'])
        indptr = np.asarray(problem['indptr'])
        shape  = tuple(problem['shape'])
        graph_csr = scipy.sparse.csr_matrix((data, indices, indptr), shape=shape)
        normed   = bool(problem.get('normed', False))
    except Exception:
        return {'laplacian': {'data': [], 'indices': [], 'indptr': [], 'shape': shape}}

    # Compute the Laplacian
    try:
        L_csr = scipy.sparse.csgraph.laplacian(graph_csr, normed=normed)
        if not isinstance(L_csr, scipy.sparse.csr_matrix):
            L_csr = L_csr.tocsr()
        L_csr.eliminate_zeros()
    except Exception:
        return {'laplacian': {'data': [], 'indices': [], 'indptr': [], 'shape': shape}}

    # Convert to lists for the expected output format
    return {
        'laplacian': {
            'data':   L_csr.data.tolist(),
            'indices':L_csr.indices.tolist(),
            'indptr': L_csr.indptr.tolist(),
            'shape':  L_csr.shape
        }
    }