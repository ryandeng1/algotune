import numpy as np

class Solver:
    def solve(self, problem: tuple) -> np.ndarray:
        """
        Compute the 2‑D correlation of two real arrays using the FFT‑based
        convolution theorem.  The correlation is computed in "full" mode,
        treating values outside the inputs as zero ("fill" boundary).

        Parameters
        ----------
        problem : tuple
            A tuple (a, b) of 2‑D numpy arrays of dtype float32 or float64.

        Returns
        -------
        numpy.ndarray
            The correlation result in full mode, dtype matching the input.
        """
        a, b = problem

        # Ensure inputs are float arrays (scipy expects float32/64)
        if a.dtype not in (np.float32, np.float64):
            a = a.astype(np.float64)
        if b.dtype not in (np.float32, np.float64):
            b = b.astype(np.float64)

        # Dimensions of full correlation output
        out_rows = a.shape[0] + b.shape[0] - 1
        out_cols = a.shape[1] + b.shape[1] - 1

        # Next power‑of‑two sizes for efficient FFT (optional but speeds up some sizes)
        def next_pow2(x): return 1 << (x - 1).bit_length()
        fft_rows = next_pow2(out_rows)
        fft_cols = next_pow2(out_cols)

        # Zero‑pad both arrays to FFT size
        fft_a = np.fft.rfftn(a, s=(fft_rows, fft_cols))
        fft_b = np.fft.rfftn(b, s=(fft_rows, fft_cols))

        # Element‑wise multiplication (correlation is convolution with flipped kernel;
        # however flipping is handled naturally by using `transpose` of b)
        prod = fft_a * fft_b.conj()  # conjugate for cross‑correlation

        # Inverse FFT to get correlation result
        corr = np.fft.irfftn(prod, s=(fft_rows, fft_cols))

        # Slice to the exact full size
        result = corr[:out_rows, :out_cols]

        # Preserve dtype of operand with higher precision
        if a.dtype == np.float64 or b.dtype == np.float64:
            result = result.astype(np.float64)
        else:
            result = result.astype(np.float32)

        return result
