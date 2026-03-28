import numpy as np

class Solver:
    def __init__(self):
        pass

    def solve(self, problem: tuple) -> np.ndarray:
        a, b = problem
        # sizes of the arrays
        ma, na = a.shape
        mb, nb = b.shape
        # full convolution size
        out_shape = (ma + mb - 1, na + nb - 1)
        # next power of two for efficient FFT (optional but can speed up)
        fft_shape = (
            1 << (out_shape[0] - 1).bit_length(),
            1 << (out_shape[1] - 1).bit_length(),
        )

        # prepare padded arrays
        fa = np.fft.fft2(a, s=fft_shape)
        # correlation uses flipped kernel, so flip b both axes
        fb = np.fft.fft2(np.flip(b), s=fft_shape)

        # element‑wise multiplication in frequency domain
        fr = fa * fb
        # inverse FFT to get full correlation result
        res = np.fft.ifft2(fr).real
        # crop to expected output size
        return res[:out_shape[0], :out_shape[1]]