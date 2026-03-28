from typing import Any, Dict, List

def _simplex_projection(y: List[float]) -> List[float]:
    """
    Project the vector `y` onto the probability simplex.
    Implements the O(n log n) algorithm from
    https://arxiv.org/pdf/1309.1541.
    """
    # sort in decreasing order
    u = sorted(y, reverse=True)
    v = 0.0
    theta = 0.0
    for i, ui in enumerate(u, start=1):
        v += ui
        # check if current threshold satisfies the condition
        if ui > (v - 1) / i:
            continue
        else:
            theta = (v - 1) / i
            break
    else:
        # all elements satisfy, theta = (v - 1)/len(u)
        theta = (v - 1) / len(u)

    return [max(yi - theta, 0.0) for yi in y]

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[float]]:
        """
        Solve the Euclidean projection onto the probability simplex.

        :param problem: A dictionary containing the key 'y' with the input vector.
        :return: A dictionary with key 'solution' containing the projected vector.
        """
        y = list(problem.get('y', []))
        solution = _simplex_projection(y)
        return {'solution': solution}