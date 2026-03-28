import numpy as np
from typing import Any

class Solver:
    # Pre‑allocate a high‑precision dtype to avoid repeated casting
    _dtype = np.float32

    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Shift a 2‑D numpy array by a (float, float) vector using an efficient FFT‑based
        interpolation.  For integer shifts the implementation falls back to np.roll
        to avoid the overhead of an FFT.

        Parameters
        ----------
        problem : dict[str, Any]
            Dictionary that must contain:
                * 'image' : 2‑D numpy array (float32 or float64)
                * 'shift' : tuple/list of two floats
            The image is shifted by the given vector and returned in the result
            dictionary under the key 'shifted_image'.

        Returns
        -------
        dict[str, Any]
            ``{'shifted_image': shifted_image}`` where ``shifted_image`` is a new
            array with the same shape as ``image``.
        """
        image = np.asarray(problem['image'], dtype=self._dtype)
        shift = np.asarray(problem['shift'], dtype=self._dtype)

        # Integer shift → fast roll
        if np.allclose(shift, np.round(shift)):
            shifted = np.roll(image, shift=tuple(np.round(shift).astype(int)), axis=(0, 1))
        else:
            # For sub‑pixel shifts use a frequency‑domain phase shift
            # Compute FFT only once
            fy, fx = np.fft.fftfreq(image.shape[0]), np.fft.fftfreq(image.shape[1])
            ky, kx = np.meshgrid(fy, fx, indexing='ij')
            phase = np.exp(-2j * np.pi * (shift[0] * ky + shift[1] * kx))
            shifted = np.real(np.fft.ifft2(np.fft.fft2(image) * phase))

        return {'shifted_image': shifted}