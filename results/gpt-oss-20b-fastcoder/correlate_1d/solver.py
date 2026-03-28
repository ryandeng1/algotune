import numpy as np

def solve(problem):
    """
    Compute the 1‑D correlation for each valid pair in the problem list.

    For mode 'valid', process only pairs where the length of the second array
    does not exceed the first.

    Parameters
    ----------
    problem : list[tuple[np.ndarray, np.ndarray]]
        List of (a, b) pairs.

    Returns
    -------
    list[np.ndarray]
        List of correlation results in the same order as the input pairs.
    """
    mode = 'valid'  # mode is fixed for this challenge
    res_list = []

    for a, b in problem:
        if mode == 'valid' and b.shape[0] > a.shape[0]:
            continue
        res_list.append(np.correlate(a, b, mode=mode))

    return res_list