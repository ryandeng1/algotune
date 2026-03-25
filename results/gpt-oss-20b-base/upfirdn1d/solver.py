import numpy as np

class Solver:
    def solve(self, problem):
        """
        Compute the upfirdn operation for each problem definition in the list.

        Parameters
        ----------
        problem : list
            A list of tuples (h, x, up, down).  Each element contains:
                * h   : 1-D array_like, filter coefficients
                * x   : 1-D array_like, input signal
                * up  : int, upsampling factor
                * down: int, downsampling factor

        Returns
        -------
        result : list
            A list of 1-D numpy arrays, each containing the result of
            the upfirdn operation for the corresponding problem.
        """
        results = []

        for h, x, up, down in problem:
            h_arr = np.asarray(h, dtype=np.float64)
            x_arr = np.asarray(x, dtype=np.float64)

            # create upsampled signal: zeros between samples
            up_len = len(x_arr) * up
            upsampled = np.zeros(up_len, dtype=np.float64)
            upsampled[::up] = x_arr

            # convolution with the filter
            conv = np.convolve(upsampled, h_arr, mode='full')

            # downsample the result
            res = conv[::down]
            results.append(res)

        return results
