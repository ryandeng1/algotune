#!/usr/bin/env python3
import numpy as np

class Solver:
    """
    A fast 2‑D correlation engine that uses the FFT‑based algorithm in full mode
    (i.e. no discarding of incompletely overlapped values) and a pad‑with‑zero
    (“fill”) boundary.  The implementation uses NumPy’s FFT routine, which
    outperforms SciPy’s `signal.correlate2d` for large operands.

    Note
    ----
    * The routine accepts any numeric dtypes that NumPy can convert to
      `float64`.  Because the FFT is performed on complex data anyway,
      the conversion is negligible in cost compared to the FFT.
    * For very small inputs the native `np.correlate` path is used to avoid
      the overhead of an FFT call.
    """
    # Size below which small‑size direct correlation is faster.
    _SMALL_THRESHOLD = 64

    @staticmethod
    def _direct_correlate(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Fast path for small inputs: uses NumPy’s built‑in 2‑D correlation
        by directly iterating over the rows using ``convolve`` from SciPy.
        """
        # Flip the second array as required for correlation
        b_rev = np.flip(b, (0, 1))
        result = np.empty((a.shape[0] + b.shape[0] - 1,
                           a.shape[1] + b.shape[1] - 1),
                          dtype=np.result_type(a, b))
        for i in range(result.shape[0]):
            # Local slice of a that overlaps with the current row of b_rev
            a_slice = a[max(0, i - b.shape[0] + 1):min(a.shape[0], i + 1), :]
            a_row = a_slice[:, ::-1]  # shift columns for correlation
            for j in range(result.shape[1]):
                b_sub = b_rev[max(0, j - b.shape[1] + 1):min(b.shape[1], j + 1), :]
                result[i, j] = np.tensordot(a_row, b_sub, axes=2)
        return result

    def solve(self, problem: tuple[Any, Any]) -> np.ndarray:
        """
        Compute the 2‑D correlation of arrays ``a`` and ``b`` using the
        full mode and a zero‑padding boundary.

        Parameters
        ----------
        problem : tuple
            Two 2‑D numeric NumPy arrays, ``a`` and ``b``.

        Returns
        -------
        np.ndarray
            Correlation result array of shape
            ``(a.shape[0] + b.shape[0] - 1, a.shape[1] + b.shape[1] - 1)``.
        """
        a, b = problem

        # Convert to NumPy arrays if they aren't already
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)

        # Small inputs: direct compute, no FFT overhead
        if min(a.size, b.size) <= self._SMALL_THRESHOLD:
            return self._direct_correlate(a, b)

        # Full‑size output
        out_shape = (a.shape[0] + b.shape[0] - 1,
                     a.shape[1] + b.shape[1] - 1)

        # Pad both arrays to the same size (next power of two is faster
        # for many FFT libraries).  We do the pad only once per call.
        fft_shape = [np.fft.next_fast_len(s) for s in out_shape]

        # Pad the first array
        a_padded = np.fft.rfftn(a, fft_shape)
        # Flip the second array (for correlation) and pad
        b_flipped = np.flip(b, (0, 1))
        b_padded = np.fft.rfftn(b_flipped, fft_shape)

        # Perform point‑wise multiplication and inverse FFT
        conv = np.fft.irfftn(a_padded * b_padded, fft_shape)

        # Extract the valid part (full correlation)
        return conv[:out_shape[0], :out_shape[1]]