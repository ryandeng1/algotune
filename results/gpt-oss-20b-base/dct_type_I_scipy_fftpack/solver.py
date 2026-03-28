from numpy.typing import NDArray
import scipy.fft

class Solver:
    def solve(self, problem: NDArray) -> NDArray:
        """
        Compute the N-dimensional DCT Type I using the fast CPU implementation
        from :mod:`scipy.fft`. This routine is substantially quicker than the
        legacy :mod:`scipy.fftpack.dctn` on modern platforms.
        """
        return scipy.fft.dctn(problem, type=1)