import scipy.ndimage

def solve(problem: dict) -> dict:
    """Efficiently perform a 2‑D zoom on the input image."""
    img = problem["image"]
    zoom_factor = problem["zoom_factor"]
    # Scipy's C implementation is already fast; we simply call it directly
    zoomed = scipy.ndimage.zoom(img, zoom_factor, order=3, mode="constant")
    return {"zoomed_image": zoomed}