from typing import Any
import scipy.ndimage

class Solver:

    def __init__(self):
        self.mode = 'constant'
        self.order = 3

    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Solves the 2D affine transformation problem using scipy.ndimage.affine_transform.

        :param problem: A dictionary representing the problem.
        :return: A dictionary with key "transformed_image":
                 "transformed_image": The transformed image as an array.
        """
        image = problem['image']
        matrix = problem['matrix']
        try:
            transformed_image = scipy.ndimage.affine_transform(image, matrix, order=self.order, mode=self.mode)
        except Exception as e:
            return {'transformed_image': []}
        else:
            pass
        finally:
            pass
        solution = {'transformed_image': transformed_image}
        return solution
