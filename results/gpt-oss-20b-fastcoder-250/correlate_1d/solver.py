import numpy as np


class Solver:
    def solve(self, problem: list) -> list:
        """
        Compute the 1D correlation for each valid pair in the problem list.

        For mode 'valid', process only pairs where the length of the second array does not exceed the first.
        Return a list of 1D arrays representing the correlation results.
        """
        mode = self.mode
        results = []

        # Use local variables for speed
        if mode == "valid":
            for a, b in problem:
                if b.ndim != 1 or a.ndim != 1:
                    continue
                if b.size > a.size:
                    continue
                # np.correlate does the same as scipy.signal.correlate for mode='valid'
                results.append(np.correlate(a, b, mode="valid"))
        else:
            for a, b in problem:
                if b.ndim != 1 or a.ndim != 1:
                    continue
                results.append(np.correlate(a, b, mode=mode))
        return results