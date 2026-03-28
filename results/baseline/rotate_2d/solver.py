from typing import Any
import scipy.ndimage

class Solver:

    def __init__(self):
        self.mode = 'constant'
        self.order = 3
        self.reshape = False

    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Solves the 2D rotation problem using scipy.ndimage.rotate.

        :param problem: A dictionary representing the problem.
        :return: A dictionary with key "rotated_image":
                 "rotated_image": The rotated image as an array.
        """
        image = problem['image']
        angle = problem['angle']
        try:
            rotated_image = scipy.ndimage.rotate(image, angle, reshape=self.reshape, order=self.order, mode=self.mode)
        except Exception as e:
            return {'rotated_image': []}
        else:
            pass
        finally:
            pass
        solution = {'rotated_image': rotated_image}
        return solution
