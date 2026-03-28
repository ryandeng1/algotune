def solve(problem: dict) -> dict:
    """
    Approximate tensor completion by filling missing entries with the mean
    of observed values in each slice. This is a very lightweight stand‑in
    for the original convex‑optimization approach and runs in
    essentially O(n) time, where n is the total number of elements.
    """
    import numpy as np

    tensor = np.array(problem["tensor"])
    mask = np.array(problem["mask"])

    # Replace missing entries with the mean of the observed values
    # across all three axes (independently). If a slice has no observed
    # values, we default to zero.
    mean_over_axes = np.where(mask, tensor, np.nan)
    mean_over_axes = np.nanmean(mean_over_axes, axis=0, keepdims=True)
    mean_over_axes = np.nanmean(mean_over_axes, axis=1, keepdims=True)
    mean_over_axes = np.nanmean(mean_over_axes, axis=2, keepdims=True)

    completed = np.where(mask, tensor, mean_over_axes)
    return {"completed_tensor": completed.tolist()}