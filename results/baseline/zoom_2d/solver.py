from typing import Any
import scipy.ndimage

class Solver:

    def __init__(self):
        self.mode = 'constant'
        self.order = 3

    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Solves the 2D zoom problem using scipy.ndimage.zoom.

        :param problem: A dictionary representing the problem.
        :return: A dictionary with key "zoomed_image":
                 "zoomed_image": The zoomed image as an array. Shape depends on zoom factor.
        """
        image = problem['image']
        zoom_factor = problem['zoom_factor']
        try:
            zoomed_image = scipy.ndimage.zoom(image, zoom_factor, order=self.order, mode=self.mode)
        except Exception as e:
            return {'zoomed_image': []}
        else:
            pass
        finally:
            pass
        solution = {'zoomed_image': zoomed_image}
        return solution
