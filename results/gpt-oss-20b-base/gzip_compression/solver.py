import gzip
from typing import Any

class Solver:
    """
    Compress a plaintext byte string with gzip using zero modification time.
    """

    def solve(self, problem: dict[str, Any]) -> dict[str, bytes]:
        """
        Compress the plaintext using the gzip algorithm with mtime=0.

        Args:
            problem: A dictionary containing a single key ``plaintext`` which
                     holds a byte string to compress.

        Returns:
            A dictionary with a single key ``compressed_data`` whose value
            is the gzip‑encoded byte string.
        """
        # Pull the plaintext directly; no error handling is required because
        # ``gzip.compress`` will raise a descriptive exception if needed.
        plaintext = problem["plaintext"]
        # The mtime parameter is only available from Python 3.8 onward.
        compressed = gzip.compress(plaintext, compresslevel=9, mtime=0)
        return {"compressed_data": compressed}