import gzip
from typing import Any, Dict

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, bytes]:
        """
        Compress the plaintext using the gzip algorithm with a fixed mtime.
        """
        plaintext = problem["plaintext"]
        # Ensure input is bytes; gzip.compress accepts bytes.
        if isinstance(plaintext, str):
            plaintext = plaintext.encode("utf-8")
        compressed_data = gzip.compress(plaintext, compresslevel=9, mtime=0)
        return {"compressed_data": compressed_data}