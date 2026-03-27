from typing import Any, Dict, List

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[float]]:
        """
        Solve the Euclidean projection onto the probability simplex problem.

        :param problem: A dictionary of the problem's parameters.
                        expected key "y" containing a list or iterable of numbers.
        :return: A dictionary with key "solution" mapping to the projected vector.
        """
        # Convert input to a flat list of floats
        y = [float(val) for val in problem.get("y", [])]
        n = len(y)
        if n == 0:
            return {"solution": []}

        # 1. Sort y in descending order
        sorted_y = sorted(y, reverse=True)

        # 2. Find the threshold rho:
        #    rho = max { j | sorted_y[j] > (sum_{i=0}^j sorted_y[i] - 1) / (j+1) }
        cumulative_sum = 0.0
        rho = -1
        for j, val in enumerate(sorted_y):
            cumulative_sum += val
            if val > (cumulative_sum - 1.0) / (j + 1):
                rho = j

        # If no such rho, the projection is trivial
        if rho == -1:
            return {"solution": [0.0] * n}

        # 3. Compute theta
        cumulative_sum_rho = sum(sorted_y[: rho + 1])
        theta = (cumulative_sum_rho - 1.0) / (rho + 1)

        # 4. Project onto the simplex
        solution = [max(yi - theta, 0.0) for yi in y]
        return {"solution": solution}