from typing import Any, Dict, List
import numpy as np
from sklearn.utils.extmath import randomized_svd

def solve(problem: Dict[str, Any]) -> Dict[str, List]:
    A = problem["matrix"]
    n_components = problem["n_components"]
    # adapt iterations based on matrix conditioning
    n_iter = 10 if problem["matrix_type"] == "ill_conditioned" else 5

    # Randomized SVD gives U, singular values, V^T
    U, s, Vt = randomized_svd(A, n_components=n_components, n_iter=n_iter, random_state=42)

    return {"U": U, "S": s, "V": Vt.T}