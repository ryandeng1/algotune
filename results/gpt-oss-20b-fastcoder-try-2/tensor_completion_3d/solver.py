import numpy as np

class Solver:
    def solve(self, problem: dict) -> dict:
        """
        Basic tensor completion by filling missing entries with zeros.
        This is a fast placeholder that runs in O(n) time.
        """
        tensor = np.array(problem.get('tensor', []))
        mask = np.array(problem.get('mask', []))
        if tensor.size == 0 or mask.size == 0:
            return {'completed_tensor': []}
        # Copy observed entries, fill missing with zeros
        completed = np.where(mask, tensor, 0)
        return {'completed_tensor': completed.tolist()}