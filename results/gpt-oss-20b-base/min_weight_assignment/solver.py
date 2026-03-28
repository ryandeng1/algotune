import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching

def solve(problem):
    try:
        mat = csr_matrix(
            (problem['data'], problem['indices'], problem['indptr']),
            shape=problem['shape'],
        )
        row_ind, col_ind = min_weight_full_bipartite_matching(mat)
        return {
            'assignment': {
                'row_ind': row_ind.tolist(),
                'col_ind': col_ind.tolist(),
            }
        }
    except Exception:
        return {'assignment': {'row_ind': [], 'col_ind': []}}