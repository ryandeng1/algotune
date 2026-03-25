import numpy as np

class Solver:
    def __init__(self, mode='full'):
        """Initialize the solver with the desired correlation mode."""
        if mode not in ('full', 'valid', 'same'):
            raise ValueError(f"Unsupported mode: {mode}")
        self.mode = mode

    def solve(self, problem: list) -> list:
        """
        Compute the 1D correlation for each pair of arrays in `problem`.

        Parameters
        ----------
        problem : list
            A list of tuples or lists where each element is a pair (a, b)
            of 1-D iterable numeric types convertible to numpy arrays.

        Returns
        -------
        list
            A list of numpy arrays containing the correlation of each pair.
            Pairs that are skipped due to the 'valid' mode rule are omitted
            from the output list.
        """
        results = []
        # Local alias for speed
        np_correlate = np.correlate
        mode = self.mode

        for pair in problem:
            if len(pair) != 2:
                continue  # skip malformed entries
            a, b = pair
            try:
                a_arr = np.asarray(a, dtype=np.float64, order='C')
                b_arr = np.asarray(b, dtype=np.float64, order='C')
            except Exception:
                continue  # skip non-numeric or non-convertible entries

            # Skip if mode is valid and b longer than a
            if mode == 'valid' and b_arr.size > a_arr.size:
                continue

            # Compute correlation
            res = np_correlate(a_arr, b_arr, mode=mode)
            results.append(res)

        return results
