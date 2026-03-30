# solver.py
import gzip
from typing import Any, Dict

class Solver:
    """
    Simple gzip compressor.
    """
    def solve(self, problem: Dict[str, Any]) -> Dict[str, bytes]:
        """
        Compress the given plaintext with gzip (compression level 9, mtime 0).

        Parameters
        ----------
        problem : dict
            Must contain the key 'plaintext' with a bytes value.

        Returns
        -------
        dict
            Mapping with a single key 'compressed_data' holding the compressed
            bytes.
        """
        # Assumes the input is already in bytes; this is the fastest path
        # to gzip compression in the stdlib.
        compressed_data = gzip.compress(problem["plaintext"], compresslevel=9, mtime=0)
        return {"compressed_data": compressed_data}